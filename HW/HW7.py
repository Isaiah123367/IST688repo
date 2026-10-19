# HW7.py
# IST 688 - Building Human-Centered AI Applications
# Homework 07: News Information Bot (for a global law firm monitoring client news)
#
# Architecture (see write-up for the reasoning):
#   OFFLINE  (python HW7.py --build)
#     news.csv -> clean + dedupe -> parse title/description/content
#              -> "legal-interest" score from transparent signal lexicons
#              -> embed with OpenAI -> persistent ChromaDB (ChromaDB_for_HW7/)
#   ONLINE   (streamlit run HW7.py)
#     Chat UI -> LLM with two tools (function calling):
#        search_news(query, company)        semantic RAG search, optional company filter
#        top_interesting_news(company, topic, n)   pre-scored candidates, LLM re-ranks + explains
#     The LLM may only report what the tools return (no outside knowledge).
#
# Secrets: put OPENAI_API_KEY in .streamlit/secrets.toml (or an env var).

# ---------------------------------------------------------------------------
# Streamlit Community Cloud ships an old sqlite3 that chromadb rejects.
# Swap in pysqlite3 before chromadb is imported (harmless if not installed).
# ---------------------------------------------------------------------------
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import re
import sys
import json
import difflib

import numpy as np
import pandas as pd
import chromadb

_HERE = os.path.dirname(os.path.abspath(__file__))


def _locate(name):
    """Find a data file whether we run from the repo root or from HW/ (as in earlier HWs)."""
    for base in (os.getcwd(), _HERE, os.path.dirname(_HERE)):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return os.path.join(os.getcwd(), name)


CSV_PATH = _locate("news.csv")
DB_PATH = _locate("ChromaDB_for_HW7")
COLLECTION = "news_hw7"
EMBED_MODEL = "text-embedding-3-small"

# Low-cost vs higher-cost model for the required comparison.
MODELS = {
    "gpt-4o-mini (low cost)": "gpt-4o-mini",
    "gpt-4o (higher cost)": "gpt-4o",
}

# ===========================================================================
# 1. DATA PREP + INTEREST SCORING (offline, deterministic, explainable)
# ===========================================================================

# Companies are often known by another name in articles.
ALIASES = {
    "Facebook": ["Meta", "Facebook", "Instagram", "WhatsApp"],
    "Alphabet": ["Alphabet", "Google", "YouTube", "Waymo"],
    "Advanced Micro Devices (AMD)": ["AMD", "Advanced Micro Devices"],
    "Taiwan Semiconductor Manufacturing Co., Ltd. (TSMC)": ["TSMC", "Taiwan Semiconductor"],
    "Hewlett Packard Enterprise": ["Hewlett Packard Enterprise", "HPE"],
    "General Motors": ["General Motors", "GM"],
    "Procter & Gamble": ["Procter & Gamble", "P&G"],
    "BBVA-Banco Bilbao Vizcaya": ["BBVA", "Banco Bilbao Vizcaya"],
    "Blackstone Inc.": ["Blackstone"],
    "The Carlyle Group": ["Carlyle"],
    "JPMorgan Chase": ["JPMorgan", "JP Morgan", "Chase"],
    "Samsung Electronics": ["Samsung"],
    "Toyota Motor": ["Toyota"],
    "Hyundai Motor": ["Hyundai"],
    "Walt Disney": ["Disney"],
    "Tata Consultancy Services": ["Tata Consultancy", "TCS"],
    "Hon Hai Precision": ["Hon Hai", "Foxconn"],
}
# first words too common to count as a mention of the company on their own
_GENERIC_FIRST = {"general", "united", "advanced", "applied", "index", "first", "national",
                  "silver", "battery", "insight", "founders", "matrix", "octopus", "thomas",
                  "hewlett", "partners", "global", "western", "taiwan", "lite-on", "quanta",
                  "analog", "texas", "thermo", "motorola", "norwest", "sapphire", "capital",
                  "walt", "tech", "ally", "jerusalem", "lighthouse", "hon"}
_SUFFIX = re.compile(
    r"\b(Inc\.?|Corporation|Corp\.?|Ltd\.?|Co\.?,?|Group|Holdings|Company)\b", re.I
)


def name_variants(company: str) -> list:
    """All the strings we accept as a mention of this company."""
    if company in ALIASES:
        return ALIASES[company]
    out = {company}
    out.update(re.findall(r"\(([^)]+)\)", company))          # parenthetical, e.g. AMD
    base = re.sub(r"\([^)]*\)", "", company)
    out.add(_SUFFIX.sub("", base).strip(" ,.-"))
    out.update(p.strip() for p in base.split("-") if p.strip())
    out.add(re.sub(r"^The ", "", company))
    first = re.sub(r"^The ", "", base).split()[0].strip(",.&") if base.split() else ""
    if len(first) >= 4 and first.lower() not in _GENERIC_FIRST:   # 'Oaktree', 'Advent'
        out.add(first)
    return [v for v in out if len(v) >= 2]


def _word_re(variants):
    parts = sorted({re.escape(v) for v in variants}, key=len, reverse=True)
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(parts) + r")(?![A-Za-z0-9])", re.I)


# Law-firm lens: what would a client-monitoring partner want to hear about?
# category: (weight, case-insensitive pattern, case-sensitive acronym pattern)
SIGNALS = {
    "litigation": (3.0,
        r"lawsuits?|sues?|sued|suing|class[- ]action|court|judge|ruling|plaintiffs?|"
        r"settle(?:s|d|ment|ments)?|appeals?|trial|litigation|verdict|injunction|"
        r"indict\w*|allegations?|alleged(?:ly)?|infring\w+|bid protests?|"
        r"protests? (?:the|an?) (?:award|contract)|"
        r"legal (?:action|options?|battles?|challenges?|fight|dispute)|unlawful\w*|illegal\w*|"
        r"violat\w+|wrongly|misled",
        r""),
    "regulatory": (3.0,
        r"antitrust|investigat\w+|probes?|probed|fines?|fined|penalt(?:y|ies)|regulators?|"
        r"regulat\w+|compliance|sanctions?|subpoena\w*|european commission|gdpr|watchdog|"
        r"monopol\w+|export controls?|tariffs?|scrutiny|crackdown|legislation|lawmakers|senators?|"
        r"tax (?:demand|notice|dispute|authorit\w+|evasion|court|tribunal)|back taxes|IRS|GST",
        r"SEC|FTC|DOJ|CFPB|CMA|FCC|FDA|EU"),
    "deals": (2.5,
        r"acquires?|acquired|acquisitions?|merger|merges?|takeover|buyout|take[- ]private|"
        r"divest\w*|spin[- ]?off|stake|ipo|takeover bid|bid for|joint venture|"
        r"agrees? to (?:buy|sell|acquire)|to acquire|buys? (?:out|back)|to sell\b[^.]{0,80}\$\d|will be sold|sold to|take[sn]? (?:\w+ ){0,3}private|"
        r"completes? (?:the )?acquisition|to sell (?:its|a|the) \w+ (?:division|unit|business)",
        r""),
    "security_privacy": (2.5,
        r"data breach|breach(?:es|ed)?|hack\w*|cyber\s?attacks?|ransomware|data leak|leaked|"
        r"vulnerabilit\w+|outage|privacy|data protection|exposed",
        r""),
    "corporate_distress": (2.0,
        r"layoffs?|laying off|lays? off|job cuts|cuts? jobs|restructur\w+|bankruptcy|insolvency|recall\w*|"
        r"profit warning|downgrad\w+|plunge[sd]?|slump\w*|probe into",
        r""),
    "large_amount": (2.0,
        r"\$\s?\d[\d.,]*\s?(?:billion|bln|bn)|\$\s?\d[\d.,]*\s?B(?![a-z])|"
        r"(?:€|£)\s?\d[\d.,]*\s?(?:billion|bn)|\d[\d.,]*\s?billion",
        r""),
    "governance": (1.5,
        r"ceo|chief executive|steps? down|resign\w*|appoints?|board|activist|shareholders?|"
        r"whistleblower|chairman",
        r""),
    "ip_ai_policy": (1.5,
        r"patents?|copyright\w*|trademarks?|trade secrets?|ban(?:s|ned)?",
        r""),
}
_COMPILED = {
    k: (w, re.compile(r"\b(?:" + ci + r")\b", re.I) if ci else None,
        re.compile(r"\b(?:" + cs + r")\b") if cs else None)
    for k, (w, ci, cs) in SIGNALS.items()
}
# Off-topic coincidences (e.g. "Andre Cisco" the NFL safety is not Cisco Systems).
NOISE = re.compile(
    r"\b(NFL Draft|Jaguars|training camp|quarterback|touchdown|linebacker|midfielder|"
    r"goalkeeper|premier league|horoscope|recipe|obituary)\b", re.I)
# Known name collisions: people/terms that merely share a client's name.
COLLISION = re.compile(
    r"(Eni Aluko|Joey Barton|Andre Cisco|Nick Chubb|Bradley Chubb|Thomas Jolly|"
    r"NEC baby formula|necrotizing enterocolitis|da Vinci|Silver Lake,? (?:East )?(?:Hollywood|"
    r"Los Angeles|neighborhood|reservoir)|Birmingham NEC|NEC Birmingham|NEC Nijmegen|Breville|"
    r"Intuit Dome)", re.I)
# Plaintiff-firm "shareholder alert" press releases: boilerplate ads, not substantive news.
SOLICITATION = re.compile(
    r"(Robbins LLP|Schall Law|Bronstein, Gewirtz|Rosen Law|Pomerantz|Levi & Korsinsky|"
    r"Faruqi|Gross Law|Portnoy Law|Bragar Eagel|investors? (?:have|has) (?:an )?opportunity|"
    r"encourages? investors|reminds? (?:investors|stockholders|shareholders)|"
    r"INVESTOR ALERT|SHAREHOLDER ALERT|stockholders? alert)", re.I)


def parse_document(doc: str):
    """The CSV packs 'Title Description: ... content: ...' into one string."""
    doc = str(doc).strip()
    title, _, rest = doc.partition(" Description:")
    desc, _, content = rest.partition(" content:")
    return title.strip(), desc.strip(), content.strip()


def score_article(company, title, desc, content):
    """Return (score, matched-signal dict, company_mentioned)."""
    head = f"{title}. {desc}"
    matched, total = {}, 0.0
    for cat, (w, ci, cs) in _COMPILED.items():
        h, b = set(), set()
        for pat in (ci, cs):
            if pat is None:
                continue
            h |= {m.group(0).lower() for m in pat.finditer(head)}
            b |= {m.group(0).lower() for m in pat.finditer(content)}
        b -= h
        n = 2 * len(h) + len(b)               # headline hits count double
        if n:
            total += w * 2 * (1 - 0.5 ** n)   # saturating: diminishing returns
            matched[cat] = sorted(h | b)[:5]
    wre = _word_re(name_variants(company))
    mentioned = bool(wre.search(f"{title} {desc} {content}"))
    in_title = bool(wre.search(title))
    total += 1.0 * in_title
    total -= 6.0 * (not mentioned)
    total -= 4.0 * bool(NOISE.search(f"{title} {desc}"))
    total -= 5.0 * bool(SOLICITATION.search(f"{title} {desc} {content[:600]}"))
    total -= 8.0 * bool(COLLISION.search(f"{title} {desc}"))
    total += 0.5 * min(len(content) / 2000, 1.0)       # substantive articles
    return round(total, 2), matched, mentioned


_STOP = set("the a an of to in on for and or with at by from as is are was be after over about into "
            "its their new says say will has have than more most this that who what how why".split())


def _stem(w):
    if len(w) > 4 and re.search(r"(xes|ches|shes|sses)$", w):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _story_tokens(title, company):
    """Title words only (summaries differ too much between outlets), minus the company's own
    name tokens, with crude plural stemming."""
    own = set(re.findall(r"[a-z0-9]+", company.lower()))
    words = re.findall(r"[a-z0-9$]+", title.lower().replace("\u2019s", "").replace("'s", ""))
    return {_stem(w) for w in words
            if w not in _STOP and w not in own and (len(w) > 2 or re.search(r"[0-9$]", w))}


def assign_stories(out: pd.DataFrame) -> pd.DataFrame:
    """Group near-duplicate articles (same company, overlapping headline words) into one
    'story'. Coverage volume is itself a signal of importance."""
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    story_of, reps = {}, {}            # reps[company] = list of (story_id, tokens)
    for i, r in out.iterrows():
        toks = _story_tokens(r["title"], r["company"])
        sid = None
        for cand_id, cand_toks in reps.setdefault(r["company"], []):
            shared = len(toks & cand_toks)
            if shared >= 3 and shared / max(1, min(len(toks), len(cand_toks))) >= 0.4:
                sid = cand_id
                break
        if sid is None:
            sid = f"{r['company']}#{len(reps[r['company']])}"
            reps[r["company"]].append((sid, toks))
        story_of[i] = sid
    out["story_id"] = pd.Series(story_of)
    out["story_size"] = out.groupby("story_id")["id"].transform("count")
    out["is_rep"] = ~out.duplicated("story_id")      # highest-scoring article represents it
    out["base_score"] = out["score"]
    out["score"] = (out["base_score"] + 1.5 * np.log2(out["story_size"])).round(2)
    return out


def prepare_dataframe(csv_path=CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["company_name"] = df["company_name"].str.strip()
    df = df.drop_duplicates(subset=["company_name", "URL"]).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        title, desc, content = parse_document(r["Document"])
        score, matched, mentioned = score_article(r["company_name"], title, desc, content)
        rows.append({
            "id": f"a{i}",
            "company": r["company_name"],
            "date": str(r["Date"])[:10],
            "title": title,
            "description": desc,
            "content": content,
            "url": r["URL"],
            "score": score,
            "signals": json.dumps(matched),
            "mentioned": mentioned,
        })
    return assign_stories(pd.DataFrame(rows))


def embed_text(row) -> str:
    body = row["content"][:1200]
    return f"{row['company']}: {row['title']}. {row['description']} {body}".strip()


# ===========================================================================
# 2. VECTOR DB (built once, outside the app)
# ===========================================================================

def get_embedding_function(api_key=None):
    from chromadb.utils import embedding_functions
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBED_MODEL)


def build_db(embedding_function, client=None, csv_path=CSV_PATH, path=DB_PATH):
    client = client or chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, embedding_function=embedding_function,
                                   metadata={"hnsw:space": "cosine"})
    df = prepare_dataframe(csv_path)
    for s in range(0, len(df), 100):
        b = df.iloc[s:s + 100]
        col.add(
            ids=b["id"].tolist(),
            documents=[embed_text(r) for _, r in b.iterrows()],
            metadatas=[{
                "company": r["company"], "date": r["date"], "title": r["title"],
                "description": r["description"][:600], "url": r["url"],
                "score": float(r["score"]), "signals": r["signals"],
                "mentioned": bool(r["mentioned"]), "story_id": r["story_id"],
                "story_size": int(r["story_size"]), "is_rep": bool(r["is_rep"]),
            } for _, r in b.iterrows()],
        )
    return col


# ===========================================================================
# 3. RETRIEVAL TOOLS (what the LLM can call)
# ===========================================================================

class NewsStore:
    def __init__(self, collection):
        self.col = collection
        got = collection.get(include=["metadatas"])
        recs = []
        for i, m in zip(got["ids"], got["metadatas"]):
            recs.append({"id": i, **m})
        self.df = pd.DataFrame(recs)
        self.companies = sorted(self.df["company"].unique())
        self._story_urls = self.df.groupby("story_id")["url"].apply(list).to_dict()
        self._res = {c: _word_re(name_variants(c)) for c in self.companies}

    # -- company resolution -------------------------------------------------
    def resolve_company(self, text):
        """Find company names mentioned in free text ('Meta' -> Facebook)."""
        if not text:
            return []
        hits = []
        for c, rx in self._res.items():
            m = rx.search(text)
            if m:
                hits.append((len(m.group(0)), c))
        if hits:
            top = max(h[0] for h in hits)
            return [c for ln, c in hits if ln == top or ln >= 4]
        close = difflib.get_close_matches(text.strip(), self.companies, n=1, cutoff=0.75)
        return close

    # -- formatting ---------------------------------------------------------
    @staticmethod
    def _fmt(m, rank=None, extra=None):
        out = {
            "company": m["company"], "date": m["date"], "title": m["title"],
            "url": m["url"], "summary": m["description"][:350],
            "interest_score": m["score"], "signals": json.loads(m["signals"]),
            "outlets_covering_story": int(m.get("story_size", 1)),
        }
        if rank is not None:
            out = {"rank": rank, **out}
        if extra:
            out.update(extra)
        return out

    def _collapse(self, rows, n):
        """One row per story (near-duplicate coverage) and per URL (an article can appear
        under several companies). Keeps the first, i.e. best-ranked, row."""
        seen_url, seen_story, out = {}, set(), []
        for r in rows:
            if r["url"] in seen_url:
                seen_url[r["url"]]["also_relates_to"].append(r["company"])
                continue
            if r["story_id"] in seen_story:
                continue
            seen_story.add(r["story_id"])
            alts = [u for u in self._story_urls.get(r["story_id"], []) if u != r["url"]][:3]
            r = dict(r, also_relates_to=[], other_coverage_urls=alts)
            seen_url[r["url"]] = r
            out.append(r)
        return out[:n]

    def _unknown_company(self, company):
        close = difflib.get_close_matches(company, self.companies, n=5, cutoff=0.4)
        return {"results": [], "candidates": [],
                "note": f"'{company}' is not one of the tracked companies. "
                        f"Closest matches: {close or 'none'}. Tell the user and ask which "
                        "company they meant; do not substitute other companies."}

    # -- tool 1: semantic search -------------------------------------------
    def search_news(self, query, company=None, n=8):
        n = max(1, min(int(n or 8), 15))
        companies = self.resolve_company(company) if company else []
        if company and not companies:
            return self._unknown_company(company)
        where = {"mentioned": True}
        if companies:
            where = {"$and": [{"mentioned": True}, {"company": {"$in": companies}}]}
        res = self.col.query(query_texts=[query], n_results=min(n * 4, 60), where=where)
        rows = []
        for m, d in zip(res["metadatas"][0], res["distances"][0]):
            rows.append(dict(m, distance=round(d, 3)))
        rows = self._collapse(rows, n)
        if not rows:
            return {"results": [], "note": f"No articles found. Known companies include: "
                                           f"{', '.join(self.companies[:25])}..."}
        return {"results": [self._fmt(r, i + 1, {"relevance_distance": r["distance"],
                                                  "also_relates_to": r["also_relates_to"],
                                                  "other_coverage_urls": r["other_coverage_urls"]})
                            for i, r in enumerate(rows)]}

    # -- tool 2: interesting-news candidates -------------------------------
    def top_interesting_news(self, company=None, topic=None, n=10):
        n = max(1, min(int(n or 10), 15))
        companies = self.resolve_company(company) if company else []
        if company and not companies:
            return self._unknown_company(company)
        df = self.df[self.df["mentioned"]]
        if companies:
            df = df[df["company"].isin(companies)]
        if topic:   # rank-based pool (robust: no embedding-distance threshold to calibrate)
            res = self.col.query(query_texts=[topic], n_results=40,
                                 where={"mentioned": True})
            keep = {m["url"] for m in res["metadatas"][0]}
            df = df[df["url"].isin(keep)]
        df = df.sort_values("score", ascending=False)
        # hand the LLM a wider pool than n so it can re-rank with judgment
        rows = self._collapse(df.to_dict("records"), min(max(n * 3, 15), 30))
        return {"instructions": f"Choose and order the best {n}. 'signals' are matched legal/"
                                "business keywords that justify interest.",
                "candidates": [self._fmt(r, i + 1, {"also_relates_to": r["also_relates_to"],
                                              "other_coverage_urls": r["other_coverage_urls"]})
                               for i, r in enumerate(rows)]}


TOOLS = [
    {"type": "function", "function": {
        "name": "search_news",
        "description": "Semantic search over the news database. Use for 'find news about X' "
                       "(a company or a topic such as antitrust, AI chips, layoffs).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Topic or question, in the user's words"},
            "company": {"type": "string", "description": "Optional company name to restrict to"},
            "n": {"type": "integer", "description": "How many articles (default 8, max 15)"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "top_interesting_news",
        "description": "Get pre-scored candidate articles for 'most interesting / most important "
                       "news' requests, optionally limited to a company and/or topic. Returns a "
                       "pool larger than n; you then re-rank and explain.",
        "parameters": {"type": "object", "properties": {
            "company": {"type": "string", "description": "Optional company name"},
            "topic": {"type": "string", "description": "Optional topic filter"},
            "n": {"type": "integer", "description": "How many to report (default 10, max 15)"}}}}},
]

SYSTEM_PROMPT = """You are a news-monitoring assistant for a large global law firm. The firm \
tracks news about its clients (large public companies and investment firms). You can ONLY \
report on the articles returned by your tools (a fixed set published 2024-07-30 to 2024-08-06). \
Never use outside knowledge to add facts; if the tools return nothing relevant, say so.

How to work:
- "Find news about X" -> call search_news. "Most interesting / important news" -> call \
top_interesting_news (use company/topic if the user gave one).
- "Interesting" means significant to a law firm advising these clients: litigation, regulatory \
or antitrust action, big M&A or financing, security/privacy incidents, leadership/governance \
turmoil, restructuring, IP/AI policy. Routine marketing, product-feature, sports or lease news \
is NOT interesting. Use the interest_score and signals as evidence, but apply judgment and \
re-order if the score misleads. Skip articles that are not really about the client \
(e.g. a person who shares a company's name).
- Rank legal/regulatory events (rulings, lawsuits, government probes, settlements, merger \
approvals, tax disputes) above market-wide stock moves, earnings, or product news. Cover the \
different companies and event types rather than several items on one theme; items with \
outlets_covering_story > 1 are bigger stories, so say so.
- In CONTEXT, state only what the summary or signals support. Do not speculate about \
consequences (e.g. "may face scrutiny") that the summary does not mention; if the summary is \
thin, say so briefly.
- For a ranked list, give for each item: rank, company, date, headline, 1-2 sentences of CONTEXT \
explaining why it matters to the firm (grounded in the summary/signals), and the URL.
- Be concise. If an article's summary is thin, say what is and isn't known.
Available companies (examples): {companies}"""


def _md(text):
    """Streamlit renders $...$ as LaTeX; escape dollar signs so '$6 billion' shows correctly."""
    return (text or "").replace("$", "\\$")


def run_chat_turn(client, model, store, history, user_msg, max_steps=4, usage=None):
    """One user turn through the tool-calling loop. Returns (answer, tool_log).
    If a dict is passed as `usage`, prompt/completion token counts are accumulated in it."""
    sys_msg = {"role": "system",
               "content": SYSTEM_PROMPT.format(companies=", ".join(store.companies))}
    msgs = [sys_msg] + history[-10:] + [{"role": "user", "content": user_msg}]  # short-term memory
    log = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(model=model, messages=msgs, tools=TOOLS,
                                              temperature=0)
        m = resp.choices[0].message
        if usage is not None and getattr(resp, "usage", None):
            usage["prompt"] = usage.get("prompt", 0) + resp.usage.prompt_tokens
            usage["completion"] = usage.get("completion", 0) + resp.usage.completion_tokens
        if not m.tool_calls:
            return m.content, log
        msgs.append({"role": "assistant", "content": m.content,
                     "tool_calls": [t.model_dump() for t in m.tool_calls]})
        for t in m.tool_calls:
            args = json.loads(t.function.arguments or "{}")
            fn = getattr(store, t.function.name, None)
            result = fn(**args) if fn else {"error": "unknown tool"}
            urls = [x["url"] for x in result.get("results", result.get("candidates", []))
                    if isinstance(x, dict) and "url" in x]
            urls += [u for x in result.get("results", result.get("candidates", []))
                     if isinstance(x, dict) for u in x.get("other_coverage_urls", [])]
            log.append({"tool": t.function.name, "args": args, "urls": urls})
            msgs.append({"role": "tool", "tool_call_id": t.id,
                         "content": json.dumps(result)})
    return "Sorry, I could not finish that request.", log


# ===========================================================================
# 4. STREAMLIT APP
# ===========================================================================

def main():
    import streamlit as st
    from openai import OpenAI

    try:                                    # may already be set by the st.navigation entrypoint
        st.set_page_config(page_title="Client News Monitor", page_icon="📰")
    except Exception:
        pass
    st.title("📰 Client News Monitor")
    st.caption("Ask about the news on our clients. Try: *Find the most interesting news* · "
               "*Find news about Microsoft* · *Any antitrust or regulatory news?*")

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:                       # no secrets.toml / key missing
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Add OPENAI_API_KEY to .streamlit/secrets.toml")
        st.stop()

    @st.cache_resource(show_spinner="Loading news database...")
    def load_store(key):
        ef = get_embedding_function(key)
        client = chromadb.PersistentClient(path=DB_PATH)
        try:
            col = client.get_collection(COLLECTION, embedding_function=ef)
        except Exception:
            col = None
        if col is None or col.count() == 0:     # missing/empty DB on a fresh deploy: rebuild
            if not os.path.exists(CSV_PATH):
                st.error(f"Could not find {CSV_PATH}. Add news.csv to the repo root.")
                st.stop()
            col = build_db(ef, client=client, csv_path=CSV_PATH)
        return NewsStore(col)

    store = load_store(api_key)
    llm = OpenAI(api_key=api_key)

    with st.sidebar:
        label = st.selectbox("Model", list(MODELS), index=0)
        show_tools = st.checkbox("Show tool calls", value=False)
        if st.button("Clear chat"):
            st.session_state.pop("messages", None)
            st.rerun()
        st.caption(f"{len(store.df)} articles · {len(store.companies)} companies")

    st.session_state.setdefault("messages", [])
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(_md(m["content"]))

    if prompt := st.chat_input("Ask about the news..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Searching the news..."):
                answer, log = run_chat_turn(llm, MODELS[label], store,
                                            st.session_state.messages, prompt)
            st.markdown(_md(answer))
            if show_tools and log:
                with st.expander("Tool calls"):
                    st.json(log)
        st.session_state.messages += [{"role": "user", "content": prompt},
                                      {"role": "assistant", "content": answer}]


def _running_in_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


if __name__ == "__main__" and "--build" in sys.argv:
    # python HW7.py --build   (reads OPENAI_API_KEY from env or .streamlit/secrets.toml)
    key = os.getenv("OPENAI_API_KEY")
    if not key and os.path.exists(".streamlit/secrets.toml"):
        m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', open(".streamlit/secrets.toml").read())
        key = m.group(1) if m else None
    col = build_db(get_embedding_function(key))
    print(f"Built {DB_PATH} with {col.count()} articles")
elif _running_in_streamlit():
    main()
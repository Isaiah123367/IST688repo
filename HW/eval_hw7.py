# eval_hw7.py - how we know the ranked news is good, and the two-LLM comparison.
#
#   python eval_hw7.py --scoring            (no API key needed)
#       Measures the pre-computed interest score against hand labels in gold_labels.csv.
#   python eval_hw7.py --llm                (needs OPENAI_API_KEY and the built ChromaDB)
#       Runs a fixed prompt set through a low-cost and a higher-cost model and writes
#       eval_results.csv (automatic metrics) + eval_answers.md (answers for human grading).
#
# gold_labels.csv: label 2 = clearly significant for a law firm monitoring client news
# (litigation, regulatory/antitrust, major M&A, security incident, governance crisis),
# 1 = moderately relevant, 0 = routine / off-topic. split=dev were used while tuning the
# keyword lexicon; split=test were labeled after drawing a fresh sample. The labels in the
# file are DRAFT labels - re-label or override them with your own judgment.

import os
import re
import sys
import json
import time

import numpy as np
import pandas as pd

import HW7

# Approximate list prices in USD per 1M tokens (input, output). CHECK the current pricing page.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}


# ---------------------------------------------------------------- metrics
def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_rels, all_rels, k):
    ideal = dcg(sorted(all_rels, reverse=True)[:k])
    return dcg(ranked_rels[:k]) / ideal if ideal else float("nan")


def eval_scoring(csv=HW7.CSV_PATH, gold=HW7._locate("gold_labels.csv")):
    df = HW7.prepare_dataframe(csv)
    art = df.drop_duplicates("url").set_index("url")
    g = pd.read_csv(gold)
    g["score"] = g["url"].map(art["score"])
    rng = np.random.default_rng(0)
    rows = []
    for name, d in [("dev", g[g.split == "dev"]), ("test", g[g.split == "test"]), ("all", g)]:
        allr = d.label.tolist()
        ranked = d.sort_values("score", ascending=False).label.tolist()
        rand = [rng.permutation(allr).tolist() for _ in range(2000)]
        rows.append({
            "split": name, "n": len(d),
            "P@10 (label=2)": round(np.mean([r == 2 for r in ranked[:10]]), 2),
            "P@10 random": round(np.mean([np.mean([r == 2 for r in p[:10]]) for p in rand]), 2),
            "NDCG@10": round(ndcg_at_k(ranked, allr, 10), 3),
            "NDCG@10 random": round(np.mean([ndcg_at_k(p, allr, 10) for p in rand]), 3),
            "Spearman": round(d.score.corr(d.label, method="spearman"), 2),
        })
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    reps = df[df.mentioned & df.is_rep].sort_values("score", ascending=False)
    l2 = g[g.label == 2].copy()
    l2["story"] = l2.url.map(art.story_id)
    for k in (10, 25, 50):
        top = set(reps.head(k).story_id)
        print(f"label-2 gold articles whose story is in the global top {k} stories: "
              f"{l2.story.isin(top).sum()} of {len(l2)}")
    return out


# ------------------------------------------------------- LLM comparison
# (prompt, tools that make sense, expect_no_results)
PROMPTS = [
    ("Find the most interesting news", {"top_interesting_news"}, False),
    ("What are the top 5 most important news stories for the firm?", {"top_interesting_news"}, False),
    ("Find news about Nvidia", {"search_news", "top_interesting_news"}, False),
    ("Find the most interesting news about Microsoft", {"top_interesting_news"}, False),
    ("Any antitrust or regulatory news?", {"search_news", "top_interesting_news"}, False),
    ("Find news about Meta", {"search_news", "top_interesting_news"}, False),
    ("What's going on with Cisco?", {"search_news", "top_interesting_news"}, False),
    ("Find the most interesting news for Coca-Cola", {"top_interesting_news"}, False),
    ("Any cybersecurity incidents affecting our clients?", {"search_news", "top_interesting_news"}, False),
    ("Find news about Eni", {"search_news", "top_interesting_news"}, False),
    ("Find news about Zzyzx Corp", {"search_news", "top_interesting_news"}, True),
    ("What is the weather in Paris?", set(), True),
]
URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def run_comparison(client, store, models, gold=HW7._locate("gold_labels.csv"), prompts=PROMPTS):
    g = pd.read_csv(gold).drop_duplicates("url").set_index("url").label
    rows, answers = [], []
    for model in models:
        for prompt, ok_tools, expect_none in prompts:
            usage, t0 = {}, time.time()
            answer, log = HW7.run_chat_turn(client, model, store, [], prompt, usage=usage)
            secs = time.time() - t0
            returned = {u for e in log for u in e.get("urls", [])}
            cited = [u.rstrip(".,;") for u in URL_RE.findall(answer or "")]
            fabricated = [u for u in cited if u not in returned]
            labeled = [g[u] for u in cited if u in g.index]
            pin, pout = PRICES.get(model, (0, 0))
            rows.append({
                "model": model, "prompt": prompt, "seconds": round(secs, 1),
                "tools_called": ",".join(e["tool"] for e in log) or "none",
                "right_tool": (not log) if not ok_tools else any(e["tool"] in ok_tools for e in log),
                "urls_cited": len(cited), "fabricated_urls": len(fabricated),
                "labeled_cited": len(labeled),
                "mean_gold_label_cited": round(float(np.mean(labeled)), 2) if labeled else None,
                "prompt_tokens": usage.get("prompt", 0), "completion_tokens": usage.get("completion", 0),
                "cost_usd": round(usage.get("prompt", 0) * pin / 1e6
                                  + usage.get("completion", 0) * pout / 1e6, 5),
                "declined_properly": (len(cited) == 0) if expect_none else None,
            })
            answers.append(f"## [{model}] {prompt}\n\n{answer}\n")
    res = pd.DataFrame(rows)
    res.to_csv("eval_results.csv", index=False)
    open("eval_answers.md", "w").write("\n".join(answers))
    summ = res.groupby("model").agg(
        avg_seconds=("seconds", "mean"), total_cost_usd=("cost_usd", "sum"),
        right_tool=("right_tool", "mean"), fabricated_urls=("fabricated_urls", "sum"),
        mean_gold_label=("mean_gold_label_cited", "mean"),
        declined_properly=("declined_properly", "mean")).round(3)
    print(summ.to_string())
    return res


if __name__ == "__main__":
    if "--scoring" in sys.argv:
        eval_scoring()
    if "--llm" in sys.argv:
        from openai import OpenAI
        key = os.getenv("OPENAI_API_KEY")
        if not key and os.path.exists(".streamlit/secrets.toml"):
            m = re.search(r'OPENAI_API_KEY\s*=\s*"([^"]+)"', open(".streamlit/secrets.toml").read())
            key = m.group(1) if m else None
        import chromadb
        ef = HW7.get_embedding_function(key)
        col = chromadb.PersistentClient(path=HW7.DB_PATH).get_collection(HW7.COLLECTION,
                                                                         embedding_function=ef)
        run_comparison(OpenAI(api_key=key), HW7.NewsStore(col), list(HW7.MODELS.values()))

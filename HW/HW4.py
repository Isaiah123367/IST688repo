# HW4.py
# IST 688 - Building Human-Centered AI Applications
# Homework 04: An iSchool Chatbot Using RAG
#
# Builds a persistent ChromaDB vector store from the provided iSchool student
# organization HTML pages, then answers questions through a Streamlit chat
# interface backed by either OpenAI or Anthropic, with a 5-interaction
# conversation memory buffer.

# ---------------------------------------------------------------------------
# Streamlit Community Cloud ships an old system sqlite3 that ChromaDB rejects.
# Swapping in pysqlite3 before chromadb is imported fixes the deploy.
# (Harmless locally - the try/except just falls through if pysqlite3 is absent.)
# ---------------------------------------------------------------------------
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import re
import glob

import streamlit as st
import chromadb
from bs4 import BeautifulSoup
from openai import OpenAI
import anthropic


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HTML_DIR = "su_orgs"                  # folder holding the unzipped HTML pages
CHROMA_PATH = "./ChromaDB_for_HW4"    # on-disk vector DB location
COLLECTION_NAME = "HW4Collection"
EMBED_MODEL = "text-embedding-3-small"
N_RESULTS = 3                         # chunks retrieved per query
MEMORY_TURNS = 5                      # last N user+assistant interactions kept

MODEL_OPTIONS = {
    "OpenAI - gpt-4o-mini (fast/cheap)": ("openai", "gpt-4o-mini"),
    "OpenAI - gpt-4o (stronger)": ("openai", "gpt-4o"),
    "Anthropic - claude-haiku-4-5 (fast/cheap)": ("anthropic", "claude-haiku-4-5-20251001"),
    "Anthropic - claude-sonnet-5 (stronger)": ("anthropic", "claude-sonnet-5"),
}


# ---------------------------------------------------------------------------
# CHUNKING STRATEGY  (Step 2.a.ii.2 of the assignment)
# ---------------------------------------------------------------------------
# Method: boundary-aware two-way split (each HTML page becomes exactly two
# mini-documents, cut at the paragraph boundary nearest the character midpoint).
#
# Why this method:
#   1. The assignment requires exactly two mini-documents per source document,
#      so fixed-size recursive chunking (which yields a variable chunk count per
#      page) is ruled out. The interesting design question is therefore *where*
#      to place the single cut.
#   2. Cutting at a paragraph boundary rather than at a raw character offset
#      keeps sentences and list items intact. A naive text[:len//2] split
#      routinely severs a sentence like "Meetings are held on Tuesdays at" /
#      "6pm in Hinds Hall," which destroys exactly the kind of specific factual
#      detail these club pages exist to convey.
#   3. Choosing the boundary *nearest the midpoint* keeps the two halves roughly
#      balanced, so neither embedding is dominated by one topic while the other
#      is a stub. Balanced chunks produce more comparable similarity scores.
#   4. These org pages are short and semantically layered - the top of a page is
#      typically identity (who the club is, its mission) and the bottom is
#      logistics (meetings, contact, how to join). Splitting in half separates
#      those two concerns into different embeddings, so a logistics question
#      retrieves the logistics half instead of an averaged, blurry whole-page
#      vector. This is the main retrieval win over storing one chunk per page.
#   5. Fallback ladder: if a page has no blank-line paragraph breaks, the splitter
#      falls back to sentence boundaries, then to word boundaries, so it never
#      fails on poorly structured markup.
#
# Trade-off worth naming: with only two chunks there is no overlap between them,
# so a fact that straddles the cut could lose surrounding context. Retrieving
# the top 3 chunks mitigates this, since both halves of a relevant page are
# usually strong matches and tend to be returned together.
# ---------------------------------------------------------------------------
def split_into_two(text: str):
    """Split text into two balanced parts at the cleanest boundary near the middle."""
    text = text.strip()
    if not text:
        return "", ""

    target = len(text) / 2

    # Try progressively finer separators until one produces a usable boundary.
    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        units = re.split(pattern, text)
        units = [u for u in units if u.strip()]
        if len(units) < 2:
            continue

        # Walk the cumulative length and pick the boundary closest to the midpoint.
        best_index, best_distance, running = 1, None, 0
        for i, unit in enumerate(units[:-1]):
            running += len(unit) + 1
            distance = abs(running - target)
            if best_distance is None or distance < best_distance:
                best_distance, best_index = distance, i + 1

        first = " ".join(units[:best_index]).strip()
        second = " ".join(units[best_index:]).strip()
        if first and second:
            return first, second

    # Single unbreakable blob - hard split as a last resort.
    midpoint = len(text) // 2
    return text[:midpoint].strip(), text[midpoint:].strip()


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------
def extract_text(path: str) -> str:
    """Pull readable text out of an HTML page, dropping script/style/nav chrome."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse the ragged whitespace that HTML extraction leaves behind, while
    # preserving blank lines so the chunker still has paragraph boundaries.
    lines = [line.strip() for line in text.splitlines()]
    cleaned, blank = [], False
    for line in lines:
        if line:
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


# ---------------------------------------------------------------------------
# Vector database
# ---------------------------------------------------------------------------
def embed_texts(openai_client, texts):
    """Embed a batch of strings with OpenAI's embedding model."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_vector_db(openai_client):
    """
    Create the ChromaDB collection only if it does not already exist.

    ChromaDB's PersistentClient writes to disk, so on a second run the
    collection is loaded back with its documents intact and the (slow, paid)
    embedding step is skipped entirely. We treat "already has documents" as the
    existence test rather than just "collection object exists," since
    get_or_create_collection would otherwise happily hand back an empty shell
    after an interrupted first build.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return collection, False  # already built - nothing to do

    html_files = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
    html_files += sorted(glob.glob(os.path.join(HTML_DIR, "*.htm")))

    if not html_files:
        st.error(
            f"No HTML files found in '{HTML_DIR}/'. "
            "Unzip the provided pages into that folder and rerun."
        )
        st.stop()

    ids, documents, metadatas = [], [], []
    progress = st.progress(0.0, text="Reading HTML pages...")

    for i, path in enumerate(html_files):
        name = os.path.basename(path)
        text = extract_text(path)
        if not text:
            continue

        first, second = split_into_two(text)
        for part_number, chunk in enumerate((first, second), start=1):
            if not chunk:
                continue
            ids.append(f"{name}::part{part_number}")
            documents.append(chunk)
            metadatas.append({"source": name, "part": part_number})

        progress.progress((i + 1) / len(html_files), text=f"Chunked {name}")

    # Embed in batches so a large folder does not blow up a single request.
    progress.progress(0.0, text="Creating embeddings...")
    batch_size = 64
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        embeddings = embed_texts(openai_client, batch_docs)
        collection.add(
            ids=ids[start:start + batch_size],
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=metadatas[start:start + batch_size],
        )
        progress.progress(
            min(1.0, (start + batch_size) / max(1, len(documents))),
            text="Creating embeddings...",
        )

    progress.empty()
    return collection, True


def retrieve_context(collection, openai_client, query: str):
    """Embed the user's question and pull back the closest chunks."""
    query_embedding = embed_texts(openai_client, [query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=N_RESULTS)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    blocks, sources = [], []
    for doc, meta in zip(documents, metadatas):
        label = f"{meta.get('source', 'unknown')} (part {meta.get('part', '?')})"
        blocks.append(f"--- Source: {label} ---\n{doc}")
        sources.append(label)
    return "\n\n".join(blocks), sources


# ---------------------------------------------------------------------------
# Conversation memory buffer
# ---------------------------------------------------------------------------
def buffered_history(messages):
    """
    Return only the last MEMORY_TURNS interactions.

    One "interaction" is a user message plus the assistant's reply, so the
    buffer is the final MEMORY_TURNS * 2 messages. The full transcript still
    lives in session_state and is still displayed - this only bounds what gets
    sent to the LLM, which keeps token cost flat over a long session while
    preserving enough context for follow-ups like "what about their dues?"
    """
    return messages[-(MEMORY_TURNS * 2):]


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a friendly assistant for the Syracuse University School of "
    "Information Studies (iSchool). You answer questions about iSchool student "
    "organizations.\n\n"
    "Rules:\n"
    "- Answer using the CONTEXT below whenever it is relevant.\n"
    "- If the context does not contain the answer, say so plainly instead of "
    "guessing. Do not invent club names, meeting times, dues, or contacts.\n"
    "- Keep answers short and concrete. Name the organization you are "
    "describing.\n"
    "- The conversation history is provided so you can resolve follow-up "
    "questions that refer back to an earlier organization."
)


def stream_openai(model, context, history):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"}]
    messages += history
    stream = client.chat.completions.create(model=model, messages=messages, stream=True)
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_anthropic(model, context, history):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}",
        messages=history,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.title("iSchool Student Organizations Chatbot")
st.caption("HW4 - RAG over the iSchool student organization pages")

with st.sidebar:
    st.header("Settings")
    choice = st.selectbox("LLM", list(MODEL_OPTIONS.keys()))
    provider, model = MODEL_OPTIONS[choice]
    st.caption(f"Memory buffer: last {MEMORY_TURNS} interactions")
    show_sources = st.checkbox("Show retrieved sources", value=True)
    if st.button("Clear conversation"):
        st.session_state.hw4_messages = []
        st.rerun()

# OpenAI client is always needed - it provides the embeddings regardless of
# which chat model is selected, so the vector DB stays consistent across models.
if "hw4_openai" not in st.session_state:
    st.session_state.hw4_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
openai_client = st.session_state.hw4_openai

# Build (or load) the vector DB once per session and cache it in session_state.
if "hw4_collection" not in st.session_state:
    with st.spinner("Preparing the vector database..."):
        collection, was_built = build_vector_db(openai_client)
    st.session_state.hw4_collection = collection
    if was_built:
        st.success(f"Vector DB created with {collection.count()} chunks.")
collection = st.session_state.hw4_collection

if "hw4_messages" not in st.session_state:
    st.session_state.hw4_messages = []

for message in st.session_state.hw4_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about iSchool student organizations..."):
    st.session_state.hw4_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the vector DB..."):
            context, sources = retrieve_context(collection, openai_client, prompt)

        history = buffered_history(st.session_state.hw4_messages)

        if provider == "openai":
            generator = stream_openai(model, context, history)
        else:
            generator = stream_anthropic(model, context, history)

        answer = st.write_stream(generator)

        if show_sources and sources:
            with st.expander("Retrieved chunks"):
                for source in sources:
                    st.write(f"- {source}")

    st.session_state.hw4_messages.append({"role": "assistant", "content": answer})

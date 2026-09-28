# HW5.py
# IST 688 - Building Human-Centered AI Applications
# Homework 05: Enhance Your Chatbot
#
# Builds on HW4 (iSchool student organizations). Instead of embedding every
# user prompt and always searching the vector DB, the LLM is given a tool,
# relevant_club_info(query). The LLM decides when a question needs the club
# documents and writes its own search query. After the vector search, the LLM
# is called again with the results in the system prompt (and no tools), the
# same way HW4 injected context. Short-term memory keeps the last 5 interactions.

# ---------------------------------------------------------------------------
# Streamlit Community Cloud ships an old sqlite3 that ChromaDB rejects.
# Swap in pysqlite3 before chromadb is imported (harmless locally).
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
import json

import streamlit as st
import chromadb
from bs4 import BeautifulSoup
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration - same vector DB as HW4, so nothing is re-embedded if it exists
# ---------------------------------------------------------------------------
HTML_DIR = "su_orgs"
CHROMA_PATH = "./ChromaDB_for_HW4"
COLLECTION_NAME = "HW4Collection"
EMBED_MODEL = "text-embedding-3-small"
N_RESULTS = 3
MEMORY_TURNS = 5

MODEL_OPTIONS = {
    "gpt-4o-mini (fast/cheap)": "gpt-4o-mini",
    "gpt-4o (stronger)": "gpt-4o",
}


# ---------------------------------------------------------------------------
# Vector DB build (same chunking and extraction as HW4). Only runs if the
# collection is empty, e.g. on a fresh Streamlit Cloud container.
# ---------------------------------------------------------------------------
def split_into_two(text: str):
    """Split text into two balanced parts at the cleanest boundary near the middle."""
    text = text.strip()
    if not text:
        return "", ""
    target = len(text) / 2
    for pattern in (r"\n\s*\n", r"(?<=[.!?])\s+", r"\s+"):
        units = [u for u in re.split(pattern, text) if u.strip()]
        if len(units) < 2:
            continue
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
    midpoint = len(text) // 2
    return text[:midpoint].strip(), text[midpoint:].strip()


def extract_text(path: str) -> str:
    """Pull readable text out of an HTML page, dropping script/style/nav chrome."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines()]
    cleaned, blank = [], False
    for line in lines:
        if line:
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


def embed_texts(openai_client, texts):
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def get_collection(openai_client):
    """Load the HW4 collection, building it first if it is empty."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    if collection.count() > 0:
        return collection

    html_files = sorted(glob.glob(os.path.join(HTML_DIR, "*.html")))
    html_files += sorted(glob.glob(os.path.join(HTML_DIR, "*.htm")))
    if not html_files:
        st.error(f"No HTML files found in '{HTML_DIR}/'. Add the org pages and rerun.")
        st.stop()

    ids, documents, metadatas = [], [], []
    for path in html_files:
        name = os.path.basename(path)
        text = extract_text(path)
        if not text:
            continue
        for part_number, chunk in enumerate(split_into_two(text), start=1):
            if chunk:
                ids.append(f"{name}::part{part_number}")
                documents.append(chunk)
                metadatas.append({"source": name, "part": part_number})

    batch_size = 64
    for start in range(0, len(documents), batch_size):
        batch_docs = documents[start:start + batch_size]
        collection.add(
            ids=ids[start:start + batch_size],
            documents=batch_docs,
            embeddings=embed_texts(openai_client, batch_docs),
            metadatas=metadatas[start:start + batch_size],
        )
    return collection


# ---------------------------------------------------------------------------
# The tool: relevant_club_info
# ---------------------------------------------------------------------------
def relevant_club_info(query: str):
    """
    Vector search over the club documents using a query written by the LLM.
    Returns the matching text (for the LLM) and source labels (for the UI).
    """
    collection = st.session_state.hw5_collection
    openai_client = st.session_state.hw5_openai

    query_embedding = embed_texts(openai_client, [query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=N_RESULTS)

    blocks, sources = [], []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        label = f"{meta.get('source', 'unknown')} (part {meta.get('part', '?')})"
        blocks.append(f"--- Source: {label} ---\n{doc}")
        sources.append(label)
    return "\n\n".join(blocks), sources


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Search the iSchool student organization documents. Use this "
                "whenever the user asks about student organizations or clubs: "
                "what they do, meetings, events, membership, dues, contacts, "
                "or recommendations. Do not use it for greetings or small talk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A standalone search query. Resolve follow-ups using "
                            "the conversation, e.g. 'When does the AI club meet?' "
                            "instead of 'when do they meet?'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Prompts and memory
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a friendly assistant for the Syracuse University School of "
    "Information Studies (iSchool). You answer questions about iSchool student "
    "organizations.\n\n"
    "Rules:\n"
    "- For any question about student organizations, call relevant_club_info "
    "to look up the answer. Never answer club questions from memory.\n"
    "- For greetings or small talk, reply briefly without calling the tool.\n"
    "- Keep answers short and concrete, and name the organization."
)

ANSWER_PROMPT = (
    "You are a friendly assistant for the Syracuse University School of "
    "Information Studies (iSchool). You answer questions about iSchool student "
    "organizations.\n\n"
    "Rules:\n"
    "- Answer using the CONTEXT below.\n"
    "- If the context does not contain the answer, say so plainly instead of "
    "guessing. Do not invent club names, meeting times, dues, or contacts.\n"
    "- Keep answers short and concrete, and name the organization.\n"
    "- Use the conversation history to understand follow-up questions."
)


def buffered_history(messages):
    """Short-term memory: only the last MEMORY_TURNS user+assistant pairs go to the LLM."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in messages[-(MEMORY_TURNS * 2):]
    ]


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------
st.title("iSchool Student Organizations Chatbot")
st.caption("HW5 - the LLM decides when to search the club documents")

with st.sidebar:
    st.header("Settings")
    model = MODEL_OPTIONS[st.selectbox("LLM", list(MODEL_OPTIONS.keys()))]
    st.caption(f"Memory buffer: last {MEMORY_TURNS} interactions")
    show_details = st.checkbox("Show tool calls and sources", value=True)
    if st.button("Clear conversation"):
        st.session_state.hw5_messages = []
        st.rerun()

if "hw5_openai" not in st.session_state:
    st.session_state.hw5_openai = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = st.session_state.hw5_openai

if "hw5_collection" not in st.session_state:
    with st.spinner("Preparing the vector database..."):
        st.session_state.hw5_collection = get_collection(client)

if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = []

for message in st.session_state.hw5_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about iSchool student organizations..."):
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = buffered_history(st.session_state.hw5_messages)

    with st.chat_message("assistant"):
        # Call 1: the LLM decides whether it needs to search the documents.
        first = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            tool_choice="auto",
        )
        reply = first.choices[0].message

        if reply.tool_calls:
            # Run the vector search for each query the LLM asked for.
            context_blocks, all_sources, queries = [], [], []
            for tool_call in reply.tool_calls:
                if tool_call.function.name != "relevant_club_info":
                    continue
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                query = args.get("query") or prompt  # fall back to the user's words
                queries.append(query)
                with st.spinner(f"Searching club info: {query}"):
                    context, sources = relevant_club_info(query)
                context_blocks.append(context)
                all_sources += sources

            # Call 2: answer with the search results in the system prompt, no tools.
            stream = client.chat.completions.create(
                model=model,
                messages=[{
                    "role": "system",
                    "content": ANSWER_PROMPT + "\n\nCONTEXT:\n" + "\n\n".join(context_blocks),
                }] + history,
                stream=True,
            )
            answer = st.write_stream(
                chunk.choices[0].delta.content
                for chunk in stream
                if chunk.choices and chunk.choices[0].delta.content
            )

            if show_details:
                with st.expander("Tool call details"):
                    for q in queries:
                        st.write(f"**relevant_club_info** query: {q}")
                    st.write("Retrieved chunks:")
                    for source in dict.fromkeys(all_sources):
                        st.write(f"- {source}")
        else:
            # No search needed (e.g. a greeting) - use the first reply directly.
            answer = reply.content or ""
            st.markdown(answer)
            if show_details:
                st.caption("No tool call - answered without searching the documents.")

    st.session_state.hw5_messages.append({"role": "assistant", "content": answer})

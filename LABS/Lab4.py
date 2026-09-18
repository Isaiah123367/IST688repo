"""
IST 688 - Lab 4: RAG Pipeline with Vector DB

Run with:  streamlit run Lab4.py
(Or add this file as a page in your multi-page app - see the integration
note at the bottom of this file.)

Required secrets (in .streamlit/secrets.toml or the Streamlit Cloud
"Secrets" settings):

    OPENAI_API_KEY = "sk-..."

Required packages (add to requirements.txt):
    streamlit
    openai
    chromadb
    pypdf
    pysqlite3-binary   # needed on Streamlit Community Cloud, see note below

Before running:
    Create a folder named "Lab4_data" next to this file and put the 7
    course syllabus PDFs in it (the same 7 files from Lab-04-Data.zip).
    Lab4.py reads every *.pdf file in that folder to build the vector DB.
"""

# --------------------------------------------------------------------------
# sqlite3 fix for Streamlit Community Cloud
# --------------------------------------------------------------------------
# Streamlit Cloud's default Linux image ships an old sqlite3 that ChromaDB
# refuses to run on. pysqlite3-binary provides a modern build; swapping it
# into sys.modules BEFORE chromadb is imported anywhere fixes this. This is
# safe to run locally too (it just no-ops if pysqlite3 isn't installed).
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import glob

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from openai import OpenAI

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Lab 4 - Course Info RAG Chatbot", page_icon="📚")
st.title("📚 Lab 4 - Course Information Chatbot (RAG)")

st.write(
    "**How this chatbot works**\n\n"
    "- On first load, every PDF in the `Lab4_data/` folder (the course "
    "syllabi) is read, embedded with OpenAI's `text-embedding-3-small`, "
    "and stored in a ChromaDB collection called **Lab4Collection**.\n"
    "- That collection is built **once per session** and cached in "
    "`st.session_state.Lab4_VectorDB`, so you don't pay to re-embed the "
    "documents every time the app reruns.\n"
    "- When you ask a question, the chatbot retrieves the **3 most "
    "relevant syllabi** from the vector DB and hands that text to the LLM "
    "(`gpt-5-mini`) as extra context. The bot will **tell you explicitly** "
    "when it's answering using that retrieved course info, versus answering "
    "from its own general knowledge."
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
# The app runs from the repo root (streamlit_app.py), but this page lives in
# LABS/, so try every plausible location for the syllabus PDFs rather than
# assuming one exact path/name.
LAB4_DATA_DIR_CANDIDATES = [
    "LABS/Lab4_data",
    "LABS/Lab-04-Data",
    "Lab4_data",
    "Lab-04-Data",
]

def resolve_lab4_data_dir() -> str:
    for candidate in LAB4_DATA_DIR_CANDIDATES:
        if glob.glob(os.path.join(candidate, "*.pdf")):
            return candidate
    # Nothing matched - return the first candidate so the error message
    # below is at least informative about where we looked.
    return LAB4_DATA_DIR_CANDIDATES[0]

LAB4_DATA_DIR = resolve_lab4_data_dir()
COLLECTION_NAME = "Lab4Collection"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5-mini"
N_RESULTS = 3
MAX_CHARS_PER_DOC_FOR_EMBEDDING = 20000  # keep well under the embedding model's token limit
MAX_CHARS_PER_DOC_IN_PROMPT = 3000       # keep the LLM prompt from ballooning

# --------------------------------------------------------------------------
# Helpers - PDF reading
# --------------------------------------------------------------------------
def extract_pdf_text(path: str) -> str:
    """Read a PDF file and return its concatenated text."""
    try:
        reader = PdfReader(path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        return f"[Error reading {path}: {e}]"


# --------------------------------------------------------------------------
# Helpers - Vector DB (Part A)
# --------------------------------------------------------------------------
def build_lab4_vectordb():
    """
    Construct (or load) the Lab4Collection ChromaDB collection:
      - one entry per PDF in LAB4_DATA_DIR
      - id / key = filename
      - document = extracted PDF text
      - metadata = {"filename": ...}
      - embeddings computed with an OpenAI embedding model
    """
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPENAI_API_KEY"],
        model_name=EMBEDDING_MODEL,
    )

    chroma_client = chromadb.Client()  # in-memory client for this session
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
    )

    pdf_paths = sorted(glob.glob(os.path.join(LAB4_DATA_DIR, "*.pdf")))

    if not pdf_paths:
        checked = ", ".join(f"'{c}/'" for c in LAB4_DATA_DIR_CANDIDATES)
        st.error(
            f"No PDF files found. Checked: {checked}. Create one of these "
            "folders (relative to the repo root, since that's where "
            "streamlit_app.py runs from) and add the 7 syllabus PDFs to it."
        )
        return collection

    ids, documents, metadatas = [], [], []
    for path in pdf_paths:
        filename = os.path.basename(path)
        text = extract_pdf_text(path)
        if not text:
            continue
        ids.append(filename)                                   # key = filename
        documents.append(text[:MAX_CHARS_PER_DOC_FOR_EMBEDDING])
        metadatas.append({"filename": filename})

    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    return collection


# --------------------------------------------------------------------------
# Helpers - Chat (Part B)
# --------------------------------------------------------------------------
def retrieve_context(collection, query: str, n_results: int = N_RESULTS):
    """Query the vector DB and return (context_text, list_of_filenames)."""
    results = collection.query(query_texts=[query], n_results=n_results)

    doc_texts = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    filenames = [m.get("filename", "unknown") for m in metadatas]

    context_blocks = []
    for filename, text in zip(filenames, doc_texts):
        snippet = text[:MAX_CHARS_PER_DOC_IN_PROMPT]
        context_blocks.append(f"--- FROM {filename} ---\n{snippet}")

    context_text = "\n\n".join(context_blocks)
    return context_text, filenames


RAG_SYSTEM_PROMPT = (
    "You are a helpful course information assistant. You have access to "
    "retrieved excerpts from course syllabi (provided below the user's "
    "question, when relevant). \n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. When the retrieved syllabus excerpts are relevant to the question, "
    "use them to answer, and explicitly say you're using information from "
    "the course syllabi (e.g., 'According to the [course name] syllabus...').\n"
    "2. If the retrieved excerpts are NOT relevant to the question, say so "
    "and answer from your own general knowledge instead, making clear "
    "that you are doing so (e.g., 'This isn't covered in the course "
    "materials I have, but in general...').\n"
    "3. Be concise and clear."
)


def stream_openai_chat(client: OpenAI, history: list, context_text: str, filenames: list):
    """Build the augmented message list and stream a gpt-5-mini response."""
    if context_text:
        retrieval_note = (
            f"\n\nRETRIEVED SYLLABUS EXCERPTS (top {len(filenames)} matches: "
            f"{', '.join(filenames)}):\n{context_text}"
        )
    else:
        retrieval_note = "\n\n(No syllabus excerpts were retrieved for this query.)"

    messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT + retrieval_note}] + history

    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# --------------------------------------------------------------------------
# Build / load the vector DB once per session
# --------------------------------------------------------------------------
if "Lab4_VectorDB" not in st.session_state:
    with st.spinner("Building Lab4Collection (embedding syllabi PDFs)..."):
        st.session_state.Lab4_VectorDB = build_lab4_vectordb()

collection = st.session_state.Lab4_VectorDB

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Options")
    st.caption(f"Vector DB: **{COLLECTION_NAME}** ({collection.count()} documents)")

    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    debug_mode = st.checkbox("🔍 Debug: test vector search (Part A)")
    if debug_mode:
        test_query = st.text_input(
            "Test search string", placeholder="e.g., Generative AI"
        )
        if test_query:
            _, test_filenames = retrieve_context(collection, test_query, n_results=3)
            st.write("Top 3 matching documents:")
            for i, fname in enumerate(test_filenames, 1):
                st.write(f"{i}. {fname}")

# --------------------------------------------------------------------------
# Conversation state
# --------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------------------------------
# Chat input / response
# --------------------------------------------------------------------------
if prompt := st.chat_input("Ask a question about the courses..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_text, retrieved_filenames = retrieve_context(collection, prompt)

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    history_for_model = st.session_state.messages  # full history; trim here if desired

    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(
                stream_openai_chat(client, history_for_model, context_text, retrieved_filenames)
            )
        except Exception as e:
            full_response = f"⚠️ Error calling OpenAI: {e}"
            st.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# --------------------------------------------------------------------------
# Multi-page app integration note
# --------------------------------------------------------------------------
# If your main app file (e.g. app.py) uses st.navigation, add Lab4 as the
# default page like this:
#
#   pages = [
#       st.Page("Lab4.py", title="Lab 4 - RAG Chatbot", default=True),
#       st.Page("Lab3.py", title="Lab 3 - Chatbot with Memory"),
#       st.Page("HW3.py", title="HW 3 - URL Chatbot"),
#       ...
#   ]
#   nav = st.navigation(pages)
#   nav.run()
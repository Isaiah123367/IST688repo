"""
IST 688 - Lab 3: Streaming Chatbot with Memory
"""

import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(page_title="Lab 3 - Chatbot with Memory", page_icon="🌐")
st.title("🌐 Lab 3 - Streaming Chatbot with Memory")

# Using explicit string concatenation here to prevent syntax errors on copy/paste
st.write(
    "**How this chatbot works**\n\n"
    "- Paste up to **two URLs** in the sidebar. Their text is scraped with "
    "`read_url_content()` and dropped into a **system prompt that is never "
    "discarded**.\n"
    "- Pick which **LLM vendor** answers your questions.\n"
    "- Responses **stream** back token-by-token as they're generated.\n"
    "- **Conversation memory:** this app keeps a rolling **buffer of the last "
    "4 messages (2 user/assistant exchanges)**. Anything older than that "
    "scrolls out of the buffer to bound token usage."
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

MEMORY_BUFFER_SIZE = 4  # last 4 messages = 2 user/assistant exchanges

@st.cache_data(show_spinner=False, ttl=3600)
def read_url_content(url: str) -> str:
    """Fetch a URL and return its readable text content."""
    if not url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Lab3 URL Chatbot)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip non-content tags
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)
        return clean_text
    except Exception as e:
        return f"[Error reading {url}: {e}]"


def build_system_prompt(url1: str, url2: str, content1: str, content2: str) -> str:
    """Build the persistent system prompt containing URL content and persona rules."""
    parts = [
        "You are a helpful assistant who answers questions using the ",
        "reference material provided below when it's relevant. If the ",
        "answer isn't in the material, say so and answer from your own ",
        "knowledge, making clear you're doing so.\n\n",
        "CRITICAL INSTRUCTIONS:\n",
        "1. You must give answers such that someone who is 10 years old can understand them.\n",
        "2. After you answer a question, you must always ask: 'Do you want more info?'\n",
        "3. If the user replies 'Yes', you must provide more information and then ask again, 'Do you want more info?'\n",
        "4. If the user replies 'No', you must go back to asking what you can help with."
    ]
    prompt_base = "".join(parts)
    
    final_parts = [prompt_base]
    if content1:
        final_parts.append(f"\n--- CONTENT FROM URL 1 ({url1}) ---\n{content1[:8000]}")
    if content2:
        final_parts.append(f"\n--- CONTENT FROM URL 2 ({url2}) ---\n{content2[:8000]}")
    if not content1 and not content2:
        final_parts.append("\n(No URL content was provided.)")
        
    return "".join(final_parts)


def get_buffered_history(messages: list) -> list:
    """Return just the most recent MEMORY_BUFFER_SIZE messages (buffer memory)."""
    return messages[-MEMORY_BUFFER_SIZE:] if len(messages) > 0 else []


def stream_openai(system_prompt: str, history: list, model: str):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    msgs = [{"role": "system", "content": system_prompt}] + history
    stream = client.chat.completions.create(
        model=model,
        messages=msgs,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_anthropic(system_prompt: str, history: list, model: str):
    client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=history,
    ) as stream:
        for text in stream.text_stream:
            yield text


# --------------------------------------------------------------------------
# Sidebar - options
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Options")

    url1 = st.text_input("URL 1", placeholder="https://example.com/article-one")
    url2 = st.text_input("URL 2 (optional)", placeholder="https://example.com/article-two")

    vendor = st.selectbox(
        "Choose an LLM vendor",
        ["OpenAI (GPT-5)", "Anthropic (Claude Opus 4.5)"],
    )

    st.caption(
        "Memory: rolling buffer of the last 4 messages "
        "(2 user/assistant exchanges)."
    )

    if st.button("🔄 Reset conversation"):
        st.session_state.messages = []
        st.rerun()

MODEL_MAP = {
    "OpenAI (GPT-5)": ("openai", "gpt-5"),
    "Anthropic (Claude Opus 4.5)": ("anthropic", "claude-opus-4-5"),
}
provider, model_name = MODEL_MAP[vendor]

# --------------------------------------------------------------------------
# Build the persistent system prompt from the URL(s)
# --------------------------------------------------------------------------
content1 = read_url_content(url1) if url1 else ""
content2 = read_url_content(url2) if url2 else ""
system_prompt = build_system_prompt(url1, url2, content1, content2)

if url1 or url2:
    with st.expander("📄 URL content loaded into the system prompt"):
        if url1:
            st.write(f"**URL 1:** {url1}")
            st.caption(f"{len(content1)} characters extracted")
        if url2:
            st.write(f"**URL 2:** {url2}")
            st.caption(f"{len(content2)} characters extracted")
else:
    st.info("Add at least one URL in the sidebar to give the chatbot context.")

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
if prompt := st.chat_input("Ask something about the URL(s)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Only the most recent buffer of messages is sent to the model,
    # alongside the always-present system prompt built above.
    history_for_model = get_buffered_history(st.session_state.messages)

    with st.chat_message("assistant"):
        try:
            if provider == "openai":
                stream_fn = stream_openai(system_prompt, history_for_model, model_name)
            else:
                stream_fn = stream_anthropic(system_prompt, history_for_model, model_name)
            full_response = st.write_stream(stream_fn)
        except Exception as e:
            full_response = f"⚠️ Error calling {vendor}: {e}"
            st.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
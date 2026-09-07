import requests
from bs4 import BeautifulSoup
import streamlit as st
from openai import OpenAI
import anthropic


def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None


st.title("🌐 HW 2: URL Summarizer (Multiple LLMs)")
st.write("Enter a URL below to get an automated summary based on your selected settings.")

# Part B: Retrieve keys strictly via Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = None

if "ANTHROPIC_API_KEY" in st.secrets:
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
else:
    anthropic_api_key = None

# URL input goes at the top of the screen (not the sidebar)
url = st.text_input("Enter a URL")

# Part C: Sidebar Options
st.sidebar.title("Summary Settings")

# Dropdown 1: Language selection
language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Spanish", "French", "German", "Chinese", "Japanese"]
)

# Dropdown 2: Summary type selection
summary_type = st.sidebar.selectbox(
    "Select Summary Type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ]
)

# LLM provider selection
st.sidebar.title("LLM Settings")
llm_provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["OpenAI", "Claude (Anthropic)"]
)

# Model selection checkbox
use_advanced = st.sidebar.checkbox("Use advanced model")

if llm_provider == "OpenAI":
    selected_model = "gpt-4o" if use_advanced else "gpt-4o-mini"
else:
    selected_model = "claude-sonnet-4-5" if use_advanced else "claude-haiku-4-5"

st.sidebar.caption(f"Using: {llm_provider} — `{selected_model}`")

# Make sure we have a valid key for the selected provider
if llm_provider == "OpenAI" and not openai_api_key:
    st.error("OpenAI API Key not found in secrets. Please configure .streamlit/secrets.toml", icon="🚨")
    st.stop()
elif llm_provider == "Claude (Anthropic)" and not anthropic_api_key:
    st.error("Anthropic API Key not found in secrets. Please configure .streamlit/secrets.toml", icon="🚨")
    st.stop()

# Main App Body
if url:
    if st.button("Generate Summary"):
        document = read_url_content(url)

        if not document:
            st.error("Could not read content from that URL. Please check the link and try again.")
        else:
            system_prompt = f"You are a helpful assistant. {summary_type}. Please write your response in {language}."

            if llm_provider == "OpenAI":
                client = OpenAI(api_key=openai_api_key)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the web page content to summarize:\n\n{document}"},
                ]
                try:
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=messages,
                        stream=True,
                    )
                    st.write_stream(stream)
                except Exception as e:
                    st.error(f"OpenAI request failed. Please check that your API key is valid. ({e})")

            else:  # Claude (Anthropic)
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                try:
                    def claude_stream():
                        with client.messages.stream(
                            model=selected_model,
                            max_tokens=1024,
                            system=system_prompt,
                            messages=[
                                {"role": "user", "content": f"Here is the web page content to summarize:\n\n{document}"}
                            ],
                        ) as stream:
                            for text in stream.text_stream:
                                yield text

                    st.write_stream(claude_stream())
                except Exception as e:
                    st.error(f"Claude request failed. Please check that your API key is valid. ({e})")

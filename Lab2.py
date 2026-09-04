import streamlit as st
from openai import OpenAI

st.title("📄 Lab 2: Document Summarizer")
st.write("Upload a document below to get an automated summary based on your selected settings.")

# Part B: Retrieve key strictly via Streamlit secrets
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    st.error("OpenAI API Key not found in secrets. Please configure .streamlit/secrets.toml", icon="🚨")
    st.stop()

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

# Model selection checkbox
use_advanced = st.sidebar.checkbox("Use advanced model")
selected_model = "gpt-4o" if use_advanced else "gpt-4o-mini"

# Main App Body
client = OpenAI(api_key=openai_api_key)
uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

if uploaded_file:
    document = uploaded_file.read().decode()
    
    if st.button("Generate Summary"):
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful assistant. {summary_type}. Please write your response in {language}."
            },
            {
                "role": "user",
                "content": f"Here is the document to summarize:\n\n{document}",
            }
        ]
        
        stream = client.chat.completions.create(
            model=selected_model,
            messages=messages,
            stream=True,
        )
        st.write_stream(stream)
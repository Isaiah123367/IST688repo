import streamlit as st
from openai import OpenAI
import pypdf

st.title("📄 Lab 2: Document Summarizer")

# Secure API key from Streamlit secrets
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Sidebar Options
st.sidebar.header("Summary Settings")
language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Spanish", "French", "German", "Chinese"]
)

summary_type = st.sidebar.selectbox(
    "Select Summary Type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points"
    ]
)

use_advanced = st.sidebar.checkbox("Use advanced model")
model_choice = "gpt-4o" if use_advanced else "gpt-4o-mini"

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Extract text from PDF
    pdf_reader = pypdf.PdfReader(uploaded_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text

    if st.button("Generate Summary"):
        if not extracted_text.strip():
            st.error("No text could be extracted from this PDF.")
        else:
            prompt = (
                f"Language: {language}\n"
                f"Task: {summary_type}\n\n"
                f"Document Content:\n{extracted_text}"
            )
            
            with st.spinner("Generating summary..."):
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                st.subheader("Summary")
                st.write(response.choices[0].message.content)

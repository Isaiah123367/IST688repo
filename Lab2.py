import streamlit as st
import pypdf
from openai import OpenAI

# Page Header
st.title("📄 Lab 2: Document Summarizer")

# Retrieve API key securely from Streamlit secrets (Part B)
api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# Sidebar Menu Options (Part C)
st.sidebar.header("Summary Settings")

# 1. Language Selection Dropdown
language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Spanish", "French", "German", "Mandarin", "Italian"]
)

# 2. Summary Format Dropdown
summary_type = st.sidebar.selectbox(
    "Select Summary Type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points"
    ]
)

# 3. Model Selection Checkbox
use_advanced = st.sidebar.checkbox("Use advanced model")

# File Uploader for PDFs
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file is not None:
    # Read text content from PDF
    reader = pypdf.PdfReader(uploaded_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + "\n"

    st.success("PDF uploaded successfully.")

    # Trigger summary generation without requiring user text input
    if st.button("Generate Summary"):
        # Select model based on checkbox
        selected_model = "gpt-4o" if use_advanced else "gpt-4o-mini"

        # Build prompt instructions
        system_prompt = f"You are a helpful assistant that summarizes documents. Always write your response in {language}."
        user_prompt = f"{summary_type}:\n\n{extracted_text}"

        with st.spinner("Summarizing..."):
            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            summary_result = response.choices[0].message.content

        st.subheader("Summary Result")
        st.write(summary_result)
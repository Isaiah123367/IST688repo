import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import io

def read_pdf_content(uploaded_file):
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return None

st.title("📄 Lab 2: Document Summarizer")
st.write("Upload a PDF below to get an automated summary based on your selected settings.")

# Part B: Retrieve key strictly via Streamlit secrets (no text input for key)
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None

if not openai_api_key:
    st.error("OpenAI API Key not found in secrets. Please configure .streamlit/secrets.toml", icon="🚨")
    st.stop()

client = OpenAI(api_key=openai_api_key)

# File upload at the top of the screen
uploaded_file = st.file_uploader("Upload a document (.pdf)", type=("pdf"))

# Part C: Sidebar options
st.sidebar.title("Summary Settings")

language = st.sidebar.selectbox(
    "Select Language",
    ["English", "Spanish", "French", "German", "Chinese", "Japanese"]
)

summary_type = st.sidebar.selectbox(
    "Select Summary Type",
    [
        "Summarize the document in 100 words",
        "Summarize the document in 2 connecting paragraphs",
        "Summarize the document in 5 bullet points",
    ]
)

st.sidebar.title("Model Settings")
use_advanced = st.sidebar.checkbox("Use advanced model")
selected_model = "gpt-4o" if use_advanced else "gpt-4o-mini"
st.sidebar.caption(f"Using: `{selected_model}`")

# Main app body
if uploaded_file:
    if st.button("Generate Summary"):
        with st.spinner("Reading document..."):
            document = read_pdf_content(uploaded_file)

        if not document:
            st.error("Could not extract text from that PDF. Please try a different file.")
        else:
            system_prompt = (
                f"You are a helpful assistant. {summary_type}. "
                f"Please write your response in {language}."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the document to summarize:\n\n{document}"},
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
else:
    st.info("Please upload a PDF to continue.")
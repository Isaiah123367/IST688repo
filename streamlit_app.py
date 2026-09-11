Streamlit app · PY
import streamlit as st
 
# --- Page definitions ---
# Paths are relative to this file's location (repo root)
 
hw1 = st.Page("HW/HW1.py", title="HW 1", icon="1️⃣")
hw2 = st.Page("HW/HW2.py", title="HW 2 - URL Summarizer", icon="2️⃣")
hw3 = st.Page("HW/HW3.py", title="HW 3 - Streaming URL Chatbot", icon="3️⃣")
 
lab1 = st.Page("LABS/Lab1.py", title="Lab 1", icon="1️⃣")
lab2 = st.Page("LABS/Lab2.py", title="Lab 2 - Document Summarizer", icon="2️⃣", default=True)
 
# --- Navigation ---
pg = st.navigation(
    {
        "HW": [hw1, hw2, hw3],
        "Labs": [lab1, lab2],
    }
)
 
pg.run()
 

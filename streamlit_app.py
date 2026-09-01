import streamlit as st

lab1_page = st.Page("Lab1.py", title="Lab 1", icon="1️⃣")
lab2_page = st.Page("Lab2.py", title="Lab 2 - Document Summarizer", icon="2️⃣", default=True)

pg = st.navigation([lab1_page, lab2_page])
pg.run()
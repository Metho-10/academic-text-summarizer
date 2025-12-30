import streamlit as st
from summarizer import summarize_text

st.set_page_config(page_title="Academic Text Summarizer", layout="centered")

st.title("Automated Academic Text Summarizer")
st.write("Summarize long academic texts using T5 Transformer")

input_text = st.text_area(
    "Paste your academic text below:",
    height=300
)

summary_length = st.slider(
    "Summary Length",
    min_value=50,
    max_value=300,
    value=150
)

if st.button("Generate Summary"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(input_text, max_length=summary_length)

        st.subheader("Summary Output")
        st.success(summary)

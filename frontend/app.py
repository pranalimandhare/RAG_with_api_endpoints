import streamlit as st
import requests

st.set_page_config(page_title="📄 PDF Intelligence Assistant", layout="wide")
st.title("📄 PDF Intelligence Assistant (Frontend)")

backend_url = "http://localhost:8000"  # adjust if deployed

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if pdf_file:
    if st.button("Read file"):
        with st.spinner("Uploading..."):
            files = {"file": pdf_file.getvalue()}
            response = requests.post(f"{backend_url}/upload-pdf", files={"file": pdf_file})
            if response.status_code == 200:
                st.success("PDF processed successfully.")
            else:
                st.error("Upload failed.")

    st.subheader("🔍 Extract Fine Prints")
    if st.button("Generate Fine Prints"):
        with st.spinner("Generating..."):
            res = requests.post(f"{backend_url}/fine-prints")
            st.write(res.json().get("answer", "Error occurred"))

    st.subheader("💬 Chat with PDF")
    question = st.text_input("Type your question:")
    if question:
        res = requests.post(f"{backend_url}/chat", json={"question": question})
        st.write("**Answer:**", res.json().get("answer", "No answer returned."))

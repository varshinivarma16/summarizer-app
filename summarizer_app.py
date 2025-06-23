import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
import docx

# App title
st.title("📄 PDF/DOCX Summarizer")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF or DOCX file", type=["pdf", "docx"])

# Extract text functions
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# When a file is uploaded
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_ext == "docx":
        text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file type.")
        text = ""

    st.success(f"✅ Extracted {len(text)} characters.")
    st.write(text[:1000])  # Show sample

    # Get desired summary length
    desired_words = st.number_input("Enter desired summary length (in words):", min_value=10, max_value=1000, value=100)
    words_in_doc = len(text.split())

    if st.button("Generate Summary"):
        if desired_words > words_in_doc:
            st.warning("⚠️ You requested more words than in the document.")
        else:
            max_len = min(1024, int(desired_words * 1.3))
            min_len = int(max_len / 3)

            st.info(f"Using max_length={max_len}, min_length={min_len}")

            # Load summarizer
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

            # Split text into chunks
            max_chunk = 1000
            chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]

            # Generate summary
            final_summary = ""
            for chunk in chunks:
                summary_part = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
                final_summary += summary_part + " "

            st.subheader("✅ Final Summary")
            st.write(final_summary)

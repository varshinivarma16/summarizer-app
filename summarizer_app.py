import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF
import docx
import io

# App title
st.title("📚 Study Buddy: Smart Summarizer + Flashcards + Quiz")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

# Extract text functions
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Model loading with caching
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_flashcard_gen():
    return pipeline("text2text-generation", model="t5-small")

summarizer = load_summarizer()
flashcard_gen = load_flashcard_gen()

def words_to_tokens(words):
    return int(words * 1.3)

# After upload
if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    if file_ext == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif file_ext == "docx":
        text = extract_text_from_docx(uploaded_file)
    elif file_ext == "txt":
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    st.success(f"✅ Extracted {len(text.split())} words.")
    st.text_area("📝 Preview of text:", value=text[:1000], height=200)

    # Controls
    desired_words = st.slider("Summary length (words)", min_value=50, max_value=280, value=150, step=10)
    num_flashcards = st.slider("Number of flashcards", min_value=1, max_value=10, value=3)
    generate_quiz = st.checkbox("Generate quiz?")
    num_quiz = st.slider("Number of quiz questions", 1, 10, 3) if generate_quiz else 0

    if st.button("Generate Study Pack"):
        max_len = words_to_tokens(desired_words)
        min_len = int(max_len / 3)
        st.info(f"Summarizing with max_length={max_len}, min_length={min_len}")

        # Split text
        chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
        final_summary = ""

        for chunk in chunks:
            try:
                summary_part = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']
                final_summary += summary_part + " "
            except Exception as e:
                st.warning(f"Skipping chunk due to error: {e}")

        st.subheader("📘 Summary")
        st.write(final_summary)

        # Flashcards
        st.subheader("🃏 Flashcards")
        short_chunks = [text[i:i+300] for i in range(0, len(text), 300)]
        flashcards = []
        for chunk in short_chunks[:num_flashcards]:
            try:
                card = flashcard_gen(f"Generate a flashcard from: {chunk}", max_length=64, do_sample=False)[0]['generated_text']
                flashcards.append(card)
                st.markdown(f"- {card}")
            except Exception as e:
                st.warning(f"Error generating flashcard: {e}")

        # Quiz
        quiz_questions = []
        if generate_quiz:
            st.subheader("❓ Quiz Questions")
            for chunk in short_chunks[:num_quiz]:
                try:
                    question = flashcard_gen(f"Generate a quiz question from: {chunk}", max_length=64, do_sample=False)[0]['generated_text']
                    quiz_questions.append(question)
                    st.markdown(f"- {question}")
                except Exception as e:
                    st.warning(f"Error generating quiz question: {e}")

        # Save to .txt
        study_pack = f"📘 Summary\n\n{final_summary.strip()}\n\n🃏 Flashcards\n\n"
        study_pack += "\n".join([f"{i+1}. {fc}" for i, fc in enumerate(flashcards)])
        if generate_quiz:
            study_pack += "\n\n❓ Quiz Questions\n\n"
            study_pack += "\n".join([f"{i+1}. {q}" for i, q in enumerate(quiz_questions)])

        st.download_button("📥 Download Study Pack (.txt)", data=study_pack, file_name="study_pack.txt", mime="text/plain")

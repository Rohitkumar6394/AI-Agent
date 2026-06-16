import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import pyttsx3
import os

st.set_page_config(page_title="Multi PDF Voice RAG Chatbot", layout="wide")

st.title("🎙️ Multi PDF RAG Voice Chatbot Agent")
st.write("PDF Summary | Voice Question | AI Voice Answer | No API Key ")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

embedding_model = load_embedding_model()
llm = load_llm()
whisper_model = load_whisper_model()

def extract_pdf_text(pdf_files):
    text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text += f"\n\n--- PDF: {pdf.name} ---\n\n"

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text

def create_chunks(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks

def create_vector_store(chunks):
    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index

def search_chunks(query, chunks, index, top_k=3):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i in indices[0]:
        if i < len(chunks):
            results.append(chunks[i])

    return results

def run_llm(prompt, max_tokens=256):
    tokenizer, model = llm

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_answer(query, context):
    prompt = f"""
Answer the question using only the given context.
If answer is not available in the context, say:
I don't know from the uploaded PDFs.

Context:
{context}

Question:
{query}

Answer:
"""

    return run_llm(prompt, max_tokens=256)

def generate_summary(text):
    small_text = text[:6000]

    prompt = f"""
Summarize this PDF content in simple language.

Give:
1. Short overview
2. Main points
3. Conclusion

PDF Content:
{small_text}

Summary:
"""

    return run_llm(prompt, max_tokens=350)

def record_voice(filename="user_voice.wav", duration=7, fs=44100):
    st.info("🎤 Recording started... speak now")

    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()
    write(filename, fs, audio)

    st.success("✅ Voice recorded")
    return filename

def voice_to_text(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.say(text)
    engine.runAndWait()

if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "summary" not in st.session_state:
    st.session_state.summary = ""

uploaded_files = st.file_uploader(
    "Upload multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("📚 Process PDFs"):
        with st.spinner("PDF files process ho rahi hain..."):
            raw_text = extract_pdf_text(uploaded_files)

            if raw_text.strip() == "":
                st.error("PDF se text extract nahi hua. Scanned/image PDF ho sakti hai.")
                st.stop()

            chunks = create_chunks(raw_text)
            index = create_vector_store(chunks)

            st.session_state.raw_text = raw_text
            st.session_state.chunks = chunks
            st.session_state.index = index

        st.success("✅ PDFs processed successfully!")

else:
    st.warning("Please upload PDF files.")

if st.session_state.raw_text:

    tab1, tab2 = st.tabs([
        "📄 PDF Summary",
        "🎙️ Voice Chat"
    ])

    with tab1:
        st.subheader("📄 PDF Summary")

        if st.button("Generate Summary"):
            with st.spinner("Summary generate ho rahi hai..."):
                st.session_state.summary = generate_summary(st.session_state.raw_text)

        if st.session_state.summary:
            st.write(st.session_state.summary)

            if st.button("🔊 Speak Summary"):
                speak_text(st.session_state.summary)

    with tab2:
        st.subheader("🎙️ User and AI Voice Communication")

        input_type = st.radio(
            "Input type choose karo:",
            ["Text", "Voice"]
        )

        user_question = ""

        if input_type == "Text":
            user_question = st.text_input("PDF se question pucho:")

        if input_type == "Voice":
            if st.button("🎤 Speak Question"):
                audio_path = record_voice(duration=7)
                user_question = voice_to_text(audio_path)

                if os.path.exists(audio_path):
                    os.remove(audio_path)

                st.success(f"🧑 You said: {user_question}")

        if user_question:
            with st.spinner("AI answer generate kar raha hai..."):
                relevant_chunks = search_chunks(
                    user_question,
                    st.session_state.chunks,
                    st.session_state.index
                )

                context = "\n\n".join(relevant_chunks)
                answer = generate_answer(user_question, context)

            st.subheader("🤖 AI Answer")
            st.write(answer)

            st.info("🔊 AI answer speaking...")
            speak_text(answer)

            with st.expander("📌 Retrieved PDF Context"):
                st.write(context)
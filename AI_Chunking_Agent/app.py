import streamlit as st
import pandas as pd
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="AI Chunking Agent", page_icon="🧠", layout="wide")

st.title("🧠 AI Chunking Agent")
st.write("PDF/TXT ko chunks mein split karo, metadata add karo, embeddings banao, aur semantic search karo.")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()


# PDF text extractor
def extract_pdf_text(file):
    reader = PdfReader(file)
    pages = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_no,
                "text": text
            })

    return pages


# TXT text extractor
def extract_txt_text(file):
    text = file.read().decode("utf-8")
    return [{"page": 1, "text": text}]


# Chunking function
def create_chunks(pages, chunk_size=500, chunk_overlap=100):
    chunks = []
    chunk_id = 1

    for page in pages:
        text = page["text"]
        page_no = page["page"]

        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_no,
                "chunk_text": chunk_text
            })

            chunk_id += 1
            start += chunk_size - chunk_overlap

    return chunks


# Create FAISS index
def create_faiss_index(chunks):
    texts = [chunk["chunk_text"] for chunk in chunks]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


# Search function
def semantic_search(query, index, chunks, top_k=3):
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "score": float(distances[0][i]),
            "chunk_id": chunks[idx]["chunk_id"],
            "page": chunks[idx]["page"],
            "chunk_text": chunks[idx]["chunk_text"]
        })

    return results


uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 300, 100)
top_k = st.sidebar.slider("Top K Results", 1, 10, 3)

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        pages = extract_pdf_text(uploaded_file)
    else:
        pages = extract_txt_text(uploaded_file)

    st.success("File loaded successfully")

    chunks = create_chunks(
        pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    st.subheader("📌 Generated Chunks")
    st.write(f"Total Chunks: {len(chunks)}")

    df = pd.DataFrame(chunks)
    st.dataframe(df)

    index, embeddings = create_faiss_index(chunks)

    st.subheader("🔍 Semantic Search")

    query = st.text_input("Ask anything from your document")

    if query:
        results = semantic_search(query, index, chunks, top_k)

        for result in results:
            st.markdown("---")
            st.write(f"### Rank {result['rank']}")
            st.write(f"**Chunk ID:** {result['chunk_id']}")
            st.write(f"**Page:** {result['page']}")
            st.write(f"**Similarity Score:** {result['score']}")
            st.write(result["chunk_text"])
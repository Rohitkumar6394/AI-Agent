from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.read_index(
    "vector_store/meeting.index"
)

with open(
    "vector_store/chunks.pkl",
    "rb"
) as f:
    chunks = pickle.load(f)

def retrieve_context(query):

    emb = model.encode([query])

    D,I = index.search(
        np.array(emb),
        k=3
    )

    context = []

    for idx in I[0]:
        context.append(chunks[idx])

    return "\n".join(context)
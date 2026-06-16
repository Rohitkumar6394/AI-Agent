from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

with open("data/meeting.txt","r",encoding="utf-8") as f:
    text = f.read()

chunks = text.split("\n")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(np.array(embeddings))

faiss.write_index(
    index,
    "vector_store/meeting.index"
)

with open(
    "vector_store/chunks.pkl",
    "wb"
) as f:
    pickle.dump(chunks,f)

print("Index Created")
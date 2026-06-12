from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading Embedding Model...")

# Free Local Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
with open("document.txt", "r", encoding="utf-8") as f:
    document = f.read()

# Split into paragraphs
chunks = document.split("\n\n")

# Create embeddings
chunk_embeddings = model.encode(chunks)

print("\nDocument Embedding Agent Ready!")
print("-" * 50)

while True:
    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Query embedding
    query_embedding = model.encode([query])

    # Similarity search
    scores = cosine_similarity(
        query_embedding,
        chunk_embeddings
    )[0]

    best_index = np.argmax(scores)

    print("\nRelevant Content:\n")
    print(chunks[best_index])

    print(
        f"\nSimilarity Score: {scores[best_index]:.4f}"
    )
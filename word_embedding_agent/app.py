from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Local embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge Base
with open("data.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

# Create embeddings
doc_embeddings = model.encode(documents)

print("Word Embedding Agent Ready!")
print("-" * 50)

while True:
    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    # Query embedding
    query_embedding = model.encode([query])

    # Similarity search
    scores = cosine_similarity(query_embedding, doc_embeddings)

    best_match = np.argmax(scores)

    print("\nMost Relevant Text:")
    print(documents[best_match])

    print("\nSimilarity Score:",
          scores[0][best_match])
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("Loading model...")

# Free local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load sentences
with open("data.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f.readlines() if line.strip()]

# Create embeddings
sentence_embeddings = model.encode(sentences)

print("\nSentence Embedding Agent Ready!")
print("-" * 50)

while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Query embedding
    query_embedding = model.encode([query])

    # Similarity calculation
    similarity_scores = cosine_similarity(
        query_embedding,
        sentence_embeddings
    )[0]

    best_index = np.argmax(similarity_scores)

    print("\nMost Similar Sentence:")
    print(sentences[best_index])

    print(f"\nSimilarity Score: {similarity_scores[best_index]:.4f}")
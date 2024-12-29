#!/usr/bin/env python3

"""
google_use_similarity.py

Example of using Google's Universal Sentence Encoder (USE) to get free embeddings
for text, then storing similarity relationships in Neo4j.

Dependencies:
    pip install py2neo tensorflow tensorflow_hub numpy
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from py2neo import Graph

# Neo4j config
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# Similarity threshold (adjust to your needs)
THRESHOLD = 0.7

def main():
    # Connect to Neo4j
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("[INFO] Connected to Neo4j.")

    # Retrieve Book nodes
    query = """
    MATCH (b:Book)
    RETURN b.id AS book_id, b.description AS book_description
    """
    books_data = graph.run(query).data()
    print(f"[INFO] Fetched {len(books_data)} Book nodes from Neo4j.")

    # Load the Universal Sentence Encoder model (this may take a few seconds on first run)
    print("[INFO] Loading Universal Sentence Encoder from TF Hub...")
    use_model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    embed = hub.load(use_model_url)

    # Prepare texts
    book_ids = [b["book_id"] for b in books_data]
    descriptions = [(b["book_description"] or "") for b in books_data]

    # Get embeddings in one batch (for all book descriptions)
    print("[INFO] Generating USE embeddings...")
    embeddings = embed(descriptions)  # shape: (num_books, embedding_dim)

    # Convert to a NumPy array for easier manipulation
    embeddings_np = embeddings.numpy()

    # Compute pairwise similarities (O(n^2) approach for demonstration)
    print("[INFO] Computing pairwise similarities...")
    pairs = []
    for i in range(len(book_ids)):
        for j in range(i+1, len(book_ids)):
            vec1 = embeddings_np[i]
            vec2 = embeddings_np[j]

            cos_sim = cosine_similarity(vec1, vec2)
            if cos_sim >= THRESHOLD:
                pairs.append((book_ids[i], book_ids[j], cos_sim))

    print(f"[INFO] Found {len(pairs)} Book pairs with similarity >= {THRESHOLD}.")

    # Create relationships in Neo4j
    print("[INFO] Creating SIMILAR_TO relationships in Neo4j...")
    for (bid1, bid2, sim) in pairs:
        graph.run(
            """
            MATCH (b1:Book {id: $bid1}), (b2:Book {id: $bid2})
            MERGE (b1)-[rel:SIMILAR_TO]->(b2)
            ON CREATE SET rel.score = $sim
            ON MATCH SET rel.score = $sim
            """,
            bid1=bid1, bid2=bid2, sim=sim
        )

    print("[DONE] Script finished successfully.")

def cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.
    """
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import time
import random
import numpy as np
from dotenv import load_dotenv
from py2neo import Graph

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

import tensorflow as tf
import tensorflow_hub as hub
from concurrent.futures import ThreadPoolExecutor

# 1. Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

load_dotenv()

THRESHOLD_OPENAI = 0.7
THRESHOLD_FLAN = 0.7
THRESHOLD_TF = 0.7

openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
flan_embedder = HuggingFaceEmbeddings(model_name="google/flan-t5-large")

USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
try:
    print("[DEBUG] Loading Universal Sentence Encoder...")
    use_model = hub.load(USE_MODEL_URL)
    print("[DEBUG] USE model loaded successfully.")
except Exception as e:
    print("[ERROR] Could not load TF Universal Sentence Encoder:", e)
    use_model = None

def get_tf_embedding(text: str) -> list:
    if use_model is None:
        return [0.0]
    return use_model([text])[0].numpy().tolist()

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

def main():
    # 1. Connect to Neo4j
    print("[INFO] Connecting to Neo4j...")
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("[INFO] Connected to Neo4j.")

    # 2. Fetch a Random Sample of 1,000 Book Nodes
    query = """
    MATCH (b:Book)
    WITH b ORDER BY rand()
    RETURN b.id AS book_id, b.description AS book_description
    LIMIT 1000
    """
    print("[INFO] Fetching a random sample of 1,000 books...")
    books_data = graph.run(query).data()
    print(f"[INFO] Fetched {len(books_data)} Book nodes.")

    # 3. Generate Embeddings with Parallel Processing
    book_embeddings_openai = {}
    book_embeddings_flan = {}
    book_embeddings_tf = {}

    def process_book(book):
        bid = book["book_id"]
        description = book["book_description"] or ""
        return {
            "id": bid,
            "openai": openai_embedder.embed_query(description),
            "flan": flan_embedder.embed_query(description),
            "tf": get_tf_embedding(description)
        }

    print("[INFO] Generating embeddings using parallel processing...")
    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_book, books_data))

    for result in results:
        book_embeddings_openai[result["id"]] = result["openai"]
        book_embeddings_flan[result["id"]] = result["flan"]
        book_embeddings_tf[result["id"]] = result["tf"]

    print("[INFO] All embeddings generated in {:.2f} seconds.".format(time.time() - start_time))

    # 4. Compute Pairwise Similarities Efficiently
    all_book_ids = [b["book_id"] for b in books_data]
    openai_pairs, flan_pairs, tf_pairs = [], [], []

    print("[INFO] Computing pairwise similarities...")
    start_time = time.time()
    for i in range(len(all_book_ids)):
        for j in range(i + 1, min(i + 50, len(all_book_ids))):  # Limit pair evaluations
            bid1 = all_book_ids[i]
            bid2 = all_book_ids[j]

            sim_openai = cosine_similarity(
                book_embeddings_openai[bid1],
                book_embeddings_openai[bid2],
            )
            if sim_openai >= THRESHOLD_OPENAI:
                openai_pairs.append((bid1, bid2, sim_openai))

            sim_flan = cosine_similarity(
                book_embeddings_flan[bid1],
                book_embeddings_flan[bid2],
            )
            if sim_flan >= THRESHOLD_FLAN:
                flan_pairs.append((bid1, bid2, sim_flan))

            sim_tf = cosine_similarity(
                book_embeddings_tf[bid1],
                book_embeddings_tf[bid2],
            )
            if sim_tf >= THRESHOLD_TF:
                tf_pairs.append((bid1, bid2, sim_tf))

    print("[INFO] Pairwise similarity computations took {:.2f} seconds.".format(time.time() - start_time))
    print(f"[INFO] Found {len(openai_pairs)} OpenAI-similar pairs.")
    print(f"[INFO] Found {len(flan_pairs)} Flan-similar pairs.")
    print(f"[INFO] Found {len(tf_pairs)} TF-similar pairs.")

    # 5. Write Relationships
    print("[INFO] Writing relationships to Neo4j...")
    for pair_type, pairs, relationship in [
        ("OpenAI", openai_pairs, "SIMILAR_TO"),
        ("Flan", flan_pairs, "SIMILAR_TO_FLAN"),
        ("TensorFlow", tf_pairs, "SIMILAR_TO_TENSORFLOW"),
    ]:
        print(f"[INFO] Writing {pair_type} relationships...")
        for (bid1, bid2, sim) in pairs:
            graph.run(
                f"""
                MATCH (b1:Book {{id: $bid1}}), (b2:Book {{id: $bid2}})
                MERGE (b1)-[rel:{relationship}]->(b2)
                ON CREATE SET rel.score = $sim
                ON MATCH SET rel.score = $sim
                """,
                bid1=bid1, bid2=bid2, sim=sim
            )

    print("[INFO] Relationship creation complete.")
    print("[DONE] Script finished successfully.")

if __name__ == "__main__":
    main()

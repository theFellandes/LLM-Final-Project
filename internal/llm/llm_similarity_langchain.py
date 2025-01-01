#!/usr/bin/env python3

import time
import numpy as np
from dotenv import load_dotenv
from py2neo import Graph

import openai
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

import tensorflow as tf
import tensorflow_hub as hub

# 1. Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

load_dotenv()

THRESHOLD_OPENAI = 0.7
THRESHOLD_FLAN = 0.7
THRESHOLD_TF = 0.7

openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# You can try smaller HF model first, e.g. "sentence-transformers/all-MiniLM-L6-v2"
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

    # 2. Fetch Book nodes
    query = """
    MATCH (b:Book)
    RETURN b.id AS book_id, b.description AS book_description
    """
    print("[INFO] Running query to fetch books...")
    books_data = graph.run(query).data()
    print(f"[INFO] Fetched {len(books_data)} Book nodes.")

    # 3. Generate embeddings
    book_embeddings_openai = {}
    book_embeddings_flan = {}
    book_embeddings_tf = {}

    print("[INFO] Generating embeddings for each Book...")
    start_time = time.time()

    for idx, book in enumerate(books_data):
        bid = book["book_id"]
        description = book["book_description"] or ""

        print(f"[DEBUG] Processing book {idx+1}/{len(books_data)} - ID: {bid}")

        # --- OpenAI Embeddings
        t0 = time.time()
        print("   [DEBUG] Generating OpenAI embedding...")
        emb_openai = openai_embedder.embed_query(description)
        book_embeddings_openai[bid] = emb_openai
        print("   [DEBUG] OpenAI embedding done in {:.2f} seconds.".format(time.time() - t0))

        # --- Flan Embeddings
        t0 = time.time()
        print("   [DEBUG] Generating Flan embedding...")
        emb_flan = flan_embedder.embed_query(description)
        book_embeddings_flan[bid] = emb_flan
        print("   [DEBUG] Flan embedding done in {:.2f} seconds.".format(time.time() - t0))

        # --- TensorFlow USE
        t0 = time.time()
        print("   [DEBUG] Generating TF embedding...")
        emb_tf = get_tf_embedding(description)
        book_embeddings_tf[bid] = emb_tf
        print("   [DEBUG] TF embedding done in {:.2f} seconds.".format(time.time() - t0))

    print("[INFO] All embeddings generated in {:.2f} seconds.".format(time.time() - start_time))

    # 4. Compute pairwise similarities
    all_book_ids = [b["book_id"] for b in books_data]
    openai_pairs, flan_pairs, tf_pairs = [], [], []

    print("[INFO] Computing pairwise similarities...")
    start_time = time.time()
    for i in range(len(all_book_ids)):
        for j in range(i + 1, len(all_book_ids)):
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

    # 5. Write relationships
    print("[INFO] Creating SIMILAR_TO (OpenAI) relationships...")
    for (bid1, bid2, sim) in openai_pairs:
        graph.run(
            """
            MATCH (b1:Book {id: $bid1}), (b2:Book {id: $bid2})
            MERGE (b1)-[rel:SIMILAR_TO]->(b2)
            ON CREATE SET rel.score = $sim
            ON MATCH SET rel.score = $sim
            """,
            bid1=bid1, bid2=bid2, sim=sim
        )

    print("[INFO] Creating SIMILAR_TO_FLAN relationships...")
    for (bid1, bid2, sim) in flan_pairs:
        graph.run(
            """
            MATCH (b1:Book {id: $bid1}), (b2:Book {id: $bid2})
            MERGE (b1)-[rel:SIMILAR_TO_FLAN]->(b2)
            ON CREATE SET rel.score = $sim
            ON MATCH SET rel.score = $sim
            """,
            bid1=bid1, bid2=bid2, sim=sim
        )

    print("[INFO] Creating SIMILAR_TO_TENSORFLOW relationships...")
    for (bid1, bid2, sim) in tf_pairs:
        graph.run(
            """
            MATCH (b1:Book {id: $bid1}), (b2:Book {id: $bid2})
            MERGE (b1)-[rel:SIMILAR_TO_TENSORFLOW]->(b2)
            ON CREATE SET rel.score = $sim
            ON MATCH SET rel.score = $sim
            """,
            bid1=bid1, bid2=bid2, sim=sim
        )

    print("[INFO] Relationship creation complete.")
    print("[DONE] Script finished successfully.")

if __name__ == "__main__":
    main()

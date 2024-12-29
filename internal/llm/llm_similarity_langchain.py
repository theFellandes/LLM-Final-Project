#!/usr/bin/env python3

"""
langchain_multi_embeddings.py

Demonstration script that:
1. Connects to Neo4j
2. Fetches Book nodes
3. Uses LangChain for embeddings from:
    - OpenAI (OpenAIEmbeddings) --> SIMILAR_TO
    - Flan-T5-Large (HuggingFaceEmbeddings) --> SIMILAR_TO_FLAN
4. Uses Universal Sentence Encoder (custom, outside LangChain) --> SIMILAR_TO_TENSORFLOW
5. Computes pairwise cosine similarities (O(n^2) for demonstration)
6. Writes relationships to Neo4j under different names

Dependencies:
    pip install py2neo openai langchain transformers tensorflow tensorflow_hub numpy
"""

import numpy as np
from dotenv import load_dotenv
from py2neo import Graph

# --- LangChain imports ---
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# --- TensorFlow Hub for Universal Sentence Encoder ---
import tensorflow as tf
import tensorflow_hub as hub

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"

# OpenAI API key
load_dotenv()

# Relationship thresholds
THRESHOLD_OPENAI = 0.7
THRESHOLD_FLAN = 0.7
THRESHOLD_TF = 0.7

# ------------------------------------------------------------------------------
# 2. Setup LangChain Embedding Objects
# ------------------------------------------------------------------------------

# 2.1. OpenAI Embeddings
openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# 2.2. Flan-T5-Large Embeddings via HuggingFaceEmbeddings
# Note: "feature-extraction" under the hood. We rely on mean pooling for a single vector.
# This may not yield as "clean" embeddings as a dedicated sentence-transformer model.
flan_embedder = HuggingFaceEmbeddings(
    model_name="google/flan-t5-large",  # or another HF checkpoint
    # Optionally specify "task='feature-extraction'" if needed, e.g.:
    # model_kwargs={"task": "feature-extraction"}
)

# ------------------------------------------------------------------------------
# 3. Setup Universal Sentence Encoder (Outside LangChain)
# ------------------------------------------------------------------------------
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
try:
    use_model = hub.load(USE_MODEL_URL)
except Exception as e:
    print("[ERROR] Could not load TF Universal Sentence Encoder:", e)
    use_model = None

def get_tf_embedding(text: str) -> list:
    """
    Retrieve embedding from the Universal Sentence Encoder (USE)
    (since LangChain doesn't currently wrap this).
    """
    if use_model is None:
        return [0.0]
    # The model can handle batches; we do single string for simplicity.
    embedding = use_model([text])
    # embedding shape: [1, 512]
    return embedding[0].numpy().tolist()

# ------------------------------------------------------------------------------
# 4. Cosine Similarity Helper
# ------------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))

# ------------------------------------------------------------------------------
# 5. Main Logic
# ------------------------------------------------------------------------------
def main():
    # 5.1 Connect to Neo4j
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("[INFO] Connected to Neo4j.")

    # 5.2 Fetch Book nodes (id, description)
    query = """
    MATCH (b:Book)
    RETURN b.id AS book_id, b.description AS book_description
    """
    books_data = graph.run(query).data()
    print(f"[INFO] Fetched {len(books_data)} Book nodes.")

    # 5.3 Generate embeddings for each pipeline
    book_embeddings_openai = {}
    book_embeddings_flan = {}
    book_embeddings_tf = {}

    print("[INFO] Generating embeddings...")

    for book in books_data:
        bid = book["book_id"]
        description = book["book_description"] or ""

        # OpenAI embeddings (LangChain)
        emb_openai = openai_embedder.embed_query(description)
        book_embeddings_openai[bid] = emb_openai

        # Flan embeddings (LangChain / HF)
        emb_flan = flan_embedder.embed_query(description)
        book_embeddings_flan[bid] = emb_flan

        # TensorFlow USE
        emb_tf = get_tf_embedding(description)
        book_embeddings_tf[bid] = emb_tf

    print("[INFO] Embeddings generated for all pipelines.")

    # 5.4 Compute pairwise similarities for each pipeline
    all_book_ids = [b["book_id"] for b in books_data]
    openai_pairs = []
    flan_pairs = []
    tf_pairs = []

    print("[INFO] Computing pairwise similarities (O(n^2))...")
    for i in range(len(all_book_ids)):
        for j in range(i + 1, len(all_book_ids)):
            bid1 = all_book_ids[i]
            bid2 = all_book_ids[j]

            # --- OpenAI similarity ---
            sim_openai = cosine_similarity(
                book_embeddings_openai[bid1],
                book_embeddings_openai[bid2],
            )
            if sim_openai >= THRESHOLD_OPENAI:
                openai_pairs.append((bid1, bid2, sim_openai))

            # --- Flan T5 Large similarity ---
            sim_flan = cosine_similarity(
                book_embeddings_flan[bid1],
                book_embeddings_flan[bid2],
            )
            if sim_flan >= THRESHOLD_FLAN:
                flan_pairs.append((bid1, bid2, sim_flan))

            # --- TF USE similarity ---
            sim_tf = cosine_similarity(
                book_embeddings_tf[bid1],
                book_embeddings_tf[bid2],
            )
            if sim_tf >= THRESHOLD_TF:
                tf_pairs.append((bid1, bid2, sim_tf))

    print(f"[INFO] Found {len(openai_pairs)} OpenAI-similar Book pairs.")
    print(f"[INFO] Found {len(flan_pairs)} Flan T5-similar Book pairs.")
    print(f"[INFO] Found {len(tf_pairs)} TF-similar Book pairs.")

    # 5.5 Write relationships into Neo4j
    #  -- OpenAI => SIMILAR_TO
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

    #  -- Flan => SIMILAR_TO_FLAN
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

    #  -- TensorFlow => SIMILAR_TO_TENSORFLOW
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

#!/usr/bin/env python3

"""
multi_embedding_similarity.py

Demonstration script that:
1. Connects to Neo4j
2. Fetches Book nodes
3. Generates embeddings using:
    - OpenAI (SIMILAR_TO)
    - Flan-T5-Large (SIMILAR_TO_FLAN)
    - TensorFlow Universal Sentence Encoder (SIMILAR_TO_TENSORFLOW)
4. Computes pairwise cosine similarities for each pipeline
5. Writes relationships to Neo4j under different relationship names

Dependencies:
    pip install py2neo openai langchain transformers tensorflow tensorflow_hub numpy

Usage:
    python multi_embedding_similarity.py
"""

import openai
import numpy as np
from py2neo import Graph

from dotenv import load_dotenv

# --- For OpenAI embeddings (LangChain) ---
from langchain.embeddings import OpenAIEmbeddings

# --- For Flan-T5-Large embeddings (Hugging Face) ---
import torch
from transformers import AutoModel, AutoTokenizer, pipeline

# --- For TensorFlow Universal Sentence Encoder (USE) ---
import tensorflow as tf
import tensorflow_hub as hub

# (Optional) If you still want Mistral, you could import or define your function and so on.
# from transformers import pipeline as mistral_pipeline
# def get_mistral_embedding(text: str) -> list:
#     ...
#     return [...]

# ------------------------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------------------------
# Neo4j connection details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# OpenAI API key
load_dotenv()

# Embedding relationship thresholds
THRESHOLD_OPENAI = 0.7
THRESHOLD_FLAN = 0.7
THRESHOLD_TF = 0.7
# THRESHOLD_MISTRAL = 0.7  # If you want Mistral, define its threshold

# ------------------------------------------------------------------------------
# 2. OpenAI Setup
# ------------------------------------------------------------------------------
openai_embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# ------------------------------------------------------------------------------
# 3. Flan-T5-Large Setup (Naive Feature Extraction)
# ------------------------------------------------------------------------------
FLAN_MODEL_NAME = "google/flan-t5-large"
try:
    # We'll use a feature-extraction pipeline, then average-pool the last hidden states
    flan_tokenizer = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME)
    flan_model = AutoModel.from_pretrained(FLAN_MODEL_NAME)
    flan_model = flan_model.to("cpu")  # or "cuda" if you have a GPU
except Exception as e:
    print("[ERROR] Could not load Flan-T5-Large model:", e)
    flan_model = None
    flan_tokenizer = None

def get_flan_embedding(text: str) -> list:
    """
    Naive function to produce embeddings from Flan-T5-Large by:
    1. Tokenizing input
    2. Running it through the model
    3. Mean-pooling the last hidden states
    """
    if flan_model is None or flan_tokenizer is None:
        return [0.0]  # fallback or raise an error

    inputs = flan_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = flan_model(**inputs)
        # outputs.last_hidden_state: shape [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.last_hidden_state.squeeze(0)  # shape [seq_len, hidden_dim]
        # Mean-pool across seq_len dimension
        embedding = hidden_states.mean(dim=0).cpu().numpy().tolist()
        return embedding

# ------------------------------------------------------------------------------
# 4. TensorFlow Universal Sentence Encoder Setup
# ------------------------------------------------------------------------------
USE_MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
try:
    use_model = hub.load(USE_MODEL_URL)
except Exception as e:
    print("[ERROR] Could not load TF Universal Sentence Encoder:", e)
    use_model = None

def get_tf_embedding(text: str) -> list:
    """
    Retrieve embedding from the Universal Sentence Encoder (USE).
    """
    if use_model is None:
        return [0.0]
    # The model can batch inputs, but let's keep it simple for one string
    embedding = use_model([text])
    # embedding is shape [1, 512]
    return embedding[0].numpy().tolist()

# ------------------------------------------------------------------------------
# (Optional) Mistral Setup
# ------------------------------------------------------------------------------
# MISTRAL_MODEL_NAME = "..."
# mistral_model = ...
# def get_mistral_embedding(text: str) -> list:
#     ...
#     return embedding_list

# ------------------------------------------------------------------------------
# 5. Cosine Similarity Helper
# ------------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    """
    v1 = np.array(vec1, dtype=np.float32)
    v2 = np.array(vec2, dtype=np.float32)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

# ------------------------------------------------------------------------------
# 6. Main Logic
# ------------------------------------------------------------------------------
def main():
    # 6.1 Connect to Neo4j
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("[INFO] Connected to Neo4j.")

    # 6.2 Fetch Book nodes
    query = """
    MATCH (b:Book)
    RETURN b.id AS book_id, b.description AS book_description
    """
    books_data = graph.run(query).data()
    print(f"[INFO] Fetched {len(books_data)} Book nodes.")

    # 6.3 Generate embeddings for each pipeline
    print("[INFO] Generating embeddings for each pipeline...")

    # We'll store all embeddings in dicts keyed by book_id
    book_embeddings_openai = {}
    book_embeddings_flan = {}
    book_embeddings_tf = {}
    # book_embeddings_mistral = {}  # If needed

    for book in books_data:
        bid = book["book_id"]
        description = book["book_description"] or ""

        # --- OpenAI ---
        emb_openai = openai_embedder.embed_query(description)
        book_embeddings_openai[bid] = emb_openai

        # --- Flan T5 Large ---
        emb_flan = get_flan_embedding(description)
        book_embeddings_flan[bid] = emb_flan

        # --- TensorFlow USE ---
        emb_tf = get_tf_embedding(description)
        book_embeddings_tf[bid] = emb_tf

        # --- Mistral (optional) ---
        # emb_mistral = get_mistral_embedding(description)
        # book_embeddings_mistral[bid] = emb_mistral

    print("[INFO] Embeddings generated.")

    # 6.4 Compute pairwise similarities for each pipeline
    all_book_ids = [b["book_id"] for b in books_data]

    # We'll collect edges for each pipeline separately
    openai_pairs = []
    flan_pairs = []
    tf_pairs = []
    # mistral_pairs = []

    # Pairwise comparison
    print("[INFO] Computing pairwise similarities...")
    for i in range(len(all_book_ids)):
        for j in range(i + 1, len(all_book_ids)):
            bid1 = all_book_ids[i]
            bid2 = all_book_ids[j]

            # --- OpenAI similarity ---
            sim_openai = cosine_similarity(
                book_embeddings_openai[bid1],
                book_embeddings_openai[bid2]
            )
            if sim_openai >= THRESHOLD_OPENAI:
                openai_pairs.append((bid1, bid2, sim_openai))

            # --- Flan similarity ---
            sim_flan = cosine_similarity(
                book_embeddings_flan[bid1],
                book_embeddings_flan[bid2]
            )
            if sim_flan >= THRESHOLD_FLAN:
                flan_pairs.append((bid1, bid2, sim_flan))

            # --- TensorFlow similarity ---
            sim_tf = cosine_similarity(
                book_embeddings_tf[bid1],
                book_embeddings_tf[bid2]
            )
            if sim_tf >= THRESHOLD_TF:
                tf_pairs.append((bid1, bid2, sim_tf))

            # --- Mistral similarity (optional) ---
            # sim_mistral = cosine_similarity(
            #     book_embeddings_mistral[bid1],
            #     book_embeddings_mistral[bid2]
            # )
            # if sim_mistral >= THRESHOLD_MISTRAL:
            #     mistral_pairs.append((bid1, bid2, sim_mistral))

    print(f"[INFO] Found {len(openai_pairs)} OpenAI-similar Book pairs (>= {THRESHOLD_OPENAI}).")
    print(f"[INFO] Found {len(flan_pairs)} Flan-similar Book pairs (>= {THRESHOLD_FLAN}).")
    print(f"[INFO] Found {len(tf_pairs)} TF-similar Book pairs (>= {THRESHOLD_TF}).")
    # print(f"[INFO] Found {len(mistral_pairs)} Mistral-similar Book pairs (>= {THRESHOLD_MISTRAL}).")

    # 6.5 Write relationships to Neo4j
    # --- OpenAI => SIMILAR_TO
    print("[INFO] Creating SIMILAR_TO (OpenAI) relationships in Neo4j...")
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

    # --- Flan => SIMILAR_TO_FLAN
    print("[INFO] Creating SIMILAR_TO_FLAN relationships in Neo4j...")
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

    # --- TensorFlow => SIMILAR_TO_TENSORFLOW
    print("[INFO] Creating SIMILAR_TO_TENSORFLOW relationships in Neo4j...")
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

    # --- Mistral => SIMILAR_TO_MISTRAL (optional)
    # print("[INFO] Creating SIMILAR_TO_MISTRAL relationships in Neo4j...")
    # for (bid1, bid2, sim) in mistral_pairs:
    #     graph.run(
    #         """
    #         MATCH (b1:Book {id: $bid1}), (b2:Book {id: $bid2})
    #         MERGE (b1)-[rel:SIMILAR_TO_MISTRAL]->(b2)
    #         ON CREATE SET rel.score = $sim
    #         ON MATCH SET rel.score = $sim
    #         """,
    #         bid1=bid1, bid2=bid2, sim=sim
    #     )

    print("[INFO] Relationship creation complete.")
    print("[DONE] Script finished successfully.")

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

import logging
import sys
import os
from knowledge_graph import KnowledgeGraph  # Assuming the KnowledgeGraph class is in knowledge_graph.py


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define Neo4j credentials and CSV directory
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password"  # Replace with your password
    books_dir = "../../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18"  # Replace with the path to your CSV files

    # Ensure the CSV directory exists
    if not os.path.exists(books_dir):
        logging.error(f"The specified CSV directory does not exist: {books_dir}")
        sys.exit(1)

    try:
        # Initialize the KnowledgeGraph instance
        logging.info("Initializing the KnowledgeGraph application...")
        kg = KnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # Start creating the graph
        logging.info(f"Processing CSV files in directory: {books_dir}")
        kg.create_graph(books_dir)

        logging.info("Knowledge graph creation completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred while creating the knowledge graph: {e}")
    finally:
        # Ensure the Neo4j connection is closed
        kg.close()
        logging.info("Neo4j connection closed.")


if __name__ == "__main__":
    main()
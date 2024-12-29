# Main Functionality
import os

from docker_reader import DockerComposeReader
from internal.graph.knowledge_graph_builder import KnowledgeGraphBuilder
from internal.db.neo4j_handler import Neo4jHandler

from dotenv import load_dotenv

from scrapper.data_scrapper import Scrapper
from scrapper.dbpedia_genre_fetcher import DBpediaHandler
from scrapper.genre_completion import GenreCompletion
from scrapper.genre_fetcher import GenreFetcher

load_dotenv()

from internal.graph.graph_llm import GraphLLM


def process_and_insert(scrapper, neo4j_handler, dbpedia_fetcher):
    graph_builder = KnowledgeGraphBuilder(neo4j_handler)

    for book_data in scrapper.read_csv_files(chunk_size=10000):
        # Fetch genre and metadata from DBpedia
        dbpedia_metadata = dbpedia_fetcher.fetch_book_metadata(book_data["name"]) or {}
        if not dbpedia_metadata.get("genre"):
            dbpedia_metadata["genre"] = GenreCompletion.complete_genre(book_data["name"], book_data.get("description"))

        # Example user data
        user_data = [{"id": "user123", "name": "Alice", "rating": 5}]

        # Build knowledge graph
        graph_builder.create_knowledge_graph(book_data, user_data, dbpedia_metadata)


if __name__ == "__main__":
    # Initialize components
    docker_compose_path = "../docker-compose.yml"
    docker_compose_reader = DockerComposeReader(yaml_file_path=docker_compose_path)
    connection_tuple = docker_compose_reader.get_neo4j_config()
    neo4j_handler = Neo4jHandler(connection_tuple)

    custom_path = os.path.abspath(
        "../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18")
    scrapper = Scrapper(base_path=custom_path)

    dbpedia_handler = DBpediaHandler()
    genre_completion = GenreCompletion()
    graph_builder = KnowledgeGraphBuilder(neo4j_handler)

    try:
        scrapper.initialize_dataset(already_installed=True)
        for book_data in scrapper.read_csv_files(chunk_size=10000):
            # Ensure book_data has an ID
            book_data = graph_builder.process_book_data(book_data)

            # Fetch metadata from DBpedia
            dbpedia_metadata = dbpedia_handler.fetch_book_metadata(book_data["name"])
            if not dbpedia_metadata or not dbpedia_metadata.get("genre"):
                # Complete genre using OpenAI if DBpedia doesn't provide it
                genre = genre_completion.complete_genre(book_data["name"], book_data.get("description"))
                book_data["genre"] = genre

            # Generate the knowledge graph
            user_data = []  # Replace with your user data
            graph_builder.create_knowledge_graph(book_data, user_data, dbpedia_metadata)
    finally:
        neo4j_handler.close()


import os

from scrapper.data_scrapper import Scrapper
from internal.db.neo4j_handler import Neo4jHandler
from sentence_transformers import SentenceTransformer, util


def add_similarity_edges(neo4j_handler, books):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    book_titles = [book["title"] for book in books]
    embeddings = model.encode(book_titles, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

    for i, book1 in enumerate(books):
        for j, book2 in enumerate(books):
            if i != j and cosine_scores[i][j] > 0.8:  # Threshold for similarity
                neo4j_handler.create_relationship(
                    {"id": book1["id"]}, {"id": book2["id"]}, "SIMILAR_TO", {"score": float(cosine_scores[i][j])}
                )


def import_data(scrapper, neo4j_handler):
    for data in scrapper.read_csv_files(chunk_size=10000):
        if "book_name" in data:  # User rating data
            neo4j_handler.create_user({"id": data["user_id"]})
            neo4j_handler.create_book({"id": data["book_name"], "title": data["book_name"]})
            neo4j_handler.create_relationship(
                {"id": data["book_name"]}, {"id": data["user_id"]}, "REVIEWED_BY", {"rating": data["rating"]}
            )
        else:  # Book data
            neo4j_handler.create_book({
                "id": data["id"],
                "title": data["name"],
                "description": data.get("description"),
                "rating": data["rating"],
                "isbn": data["isbn"]
            })
            if data.get("authors"):
                authors = data["authors"].split(", ")
                for author in authors:
                    neo4j_handler.create_author({"name": author})
                    neo4j_handler.create_relationship({"id": data["id"]}, {"id": author}, "WRITTEN_BY")
            if data.get("genres"):
                genres = data["genres"].split(", ")
                for genre in genres:
                    neo4j_handler.create_genre({"name": genre})
                    neo4j_handler.create_relationship({"id": data["id"]}, {"id": genre}, "BELONGS_TO")


def main():
    # Initialize Scrapper and Neo4jHandler
    custom_path = os.path.abspath(
        "../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18")
    scrapper = Scrapper(base_path=custom_path)
    neo4j_handler = Neo4jHandler(uri="bolt://localhost:7687", user="neo4j", password="password")

    try:
        # Import data
        scrapper.initialize_dataset(already_installed=True)
        import_data(scrapper, neo4j_handler)

        # Add semantic similarity
        books = []  # Collect book data to calculate similarity
        for book in scrapper.read_csv_files(chunk_size=10000):
            if "id" in book:
                books.append(book)
        add_similarity_edges(neo4j_handler, books)
    finally:
        neo4j_handler.close()

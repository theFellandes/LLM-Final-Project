import uuid
import logging

logging.basicConfig(level=logging.INFO)


class KnowledgeGraphBuilder:
    def __init__(self, neo4j_handler):
        self.neo4j_handler = neo4j_handler

    @staticmethod
    def process_book_data(book_data):
        """
        Ensure the book data has a unique ID. Generate one if missing.
        """
        if "id" not in book_data or not book_data["id"]:
            book_data["id"] = str(uuid.uuid4())  # Generate a unique ID
        return book_data

    def create_book_node(self, book_data, dbpedia_metadata):
        """
        Create or update a book node with enriched metadata.
        """
        dbpedia_metadata = dbpedia_metadata or {}
        logging.info(f"Creating book node with data: {book_data}, DBpedia metadata: {dbpedia_metadata}")

        book_data["id"] = book_data.get("id") or str(uuid.uuid4())
        book_node = {
            "label": "Book",
            "properties": {
                "id": book_data["id"],
                "title": book_data["name"],
                "description": dbpedia_metadata.get("description"),
                "genre": dbpedia_metadata.get("genre") or book_data.get("genre"),
                "isbn": book_data.get("isbn"),
                "rating": book_data.get("rating"),
                "pagesNumber": book_data.get("pages_number"),
                "publisher": book_data.get("publisher"),
                "publishYear": book_data.get("publish_year")
            }
        }
        self.neo4j_handler.create_node(book_node["label"], book_node["properties"])
        return book_node

    def create_related_nodes_and_relationships(self, book_node, author_name, genre_name, user_data):
        """
        Create related nodes (Author, Genre, User) and relationships for the book.
        """
        # Author Node
        if author_name:
            author_node = {"label": "Author", "properties": {"id": str(uuid.uuid4()), "name": author_name}}
            self.neo4j_handler.create_node(author_node["label"], author_node["properties"])
            self.neo4j_handler.create_relationship(book_node["properties"], author_node["properties"], "WRITTEN_BY")

        # Genre Node
        if genre_name:
            genre_node = {"label": "Genre", "properties": {"id": str(uuid.uuid4()), "name": genre_name}}
            self.neo4j_handler.create_node(genre_node["label"], genre_node["properties"])
            self.neo4j_handler.create_relationship(book_node["properties"], genre_node["properties"], "BELONGS_TO")

        # User Nodes and Reviews
        for user in user_data:
            user_node = {"label": "User", "properties": {"id": user["id"], "name": user["name"]}}
            self.neo4j_handler.create_node(user_node["label"], user_node["properties"])
            self.neo4j_handler.create_relationship(
                book_node["properties"], user_node["properties"], "REVIEWED_BY", {"rating": user["rating"]}
            )

    def create_knowledge_graph(self, book_data, user_data, dbpedia_metadata):
        """
        Create the complete knowledge graph for a book.
        """
        dbpedia_metadata = dbpedia_metadata or {}  # Ensure it's not None
        book_node = self.create_book_node(book_data, dbpedia_metadata)
        self.create_related_nodes_and_relationships(
            book_node,
            dbpedia_metadata.get("author"),
            dbpedia_metadata.get("genre") or book_data.get("genre"),
            user_data
        )



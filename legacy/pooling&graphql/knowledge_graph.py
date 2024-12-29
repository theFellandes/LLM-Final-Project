import os
import uuid
import logging
from SPARQLWrapper import SPARQLWrapper, JSON
from neo4j import GraphDatabase


class KnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_graph(self, books_dir):
        sparql = SPARQLWrapper("https://dbpedia.org/sparql")

        with self.driver.session() as session:
            try:
                # Process each CSV file in the directory
                for file_name in os.listdir(books_dir):
                    file_path = os.path.join(books_dir, file_name)
                    if "book" in file_name:
                        books_df = self.load_csv(file_path)
                        for _, row in books_df.iterrows():
                            book_id = str(uuid.uuid4())
                            session.write_transaction(self._create_book, book_id, row)

                            # Create author relations
                            authors = row['Authors'].split(",")
                            for author in authors:
                                session.write_transaction(self._create_author_relation, book_id, author.strip())

                            # Fetch and create genre relations
                            genres = self.get_genres_from_dbpedia(sparql, row['ISBN'])
                            for genre in genres:
                                session.write_transaction(self._create_genre_relation, book_id, genre)

                    elif "user_rating" in file_name:
                        ratings_df = self.load_csv(file_path)
                        for _, row in ratings_df.iterrows():
                            user_id = f"user_{row['ID']}"
                            session.write_transaction(self._create_user, user_id)
                            session.write_transaction(self._create_review_relation, row['Name'], user_id)
            except Exception as e:
                logging.error(f"Failed to create graph: {e}")
                raise

    def load_csv(self, file_path):
        import pandas as pd
        return pd.read_csv(file_path)

    def get_genres_from_dbpedia(self, sparql, isbn):
        query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dct: <http://purl.org/dc/terms/>

        SELECT ?genre
        WHERE {{
          ?book rdf:type dbo:Book .
          OPTIONAL {{ ?book dbo:isbn ?isbn FILTER (?isbn = "{isbn}") }}
          OPTIONAL {{ ?book dct:subject/rdfs:label ?genre FILTER (lang(?genre) = "en") }}
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)

        try:
            results = sparql.query().convert()
            genres = []
            if "results" in results and "bindings" in results["results"]:
                for result in results["results"]["bindings"]:
                    genres.append(result["genre"]["value"])
            return genres
        except Exception as e:
            logging.error(f"Error querying DBpedia for ISBN {isbn}: {e}")
            return []

    # Neo4j transaction functions
    def _create_book(self, tx, book_id, row):
        tx.run(
            "MERGE (b:Book {id: $id, name: $name, isbn: $isbn})",
            id=book_id,
            name=row['Name'],
            isbn=row['ISBN']
        )

    def _create_author_relation(self, tx, book_id, author_name):
        tx.run(
            "MERGE (a:Author {name: $name}) "
            "MERGE (b:Book {id: $book_id}) "
            "MERGE (b)-[:WRITTEN_BY]->(a)",
            name=author_name,
            book_id=book_id
        )

    def _create_genre_relation(self, tx, book_id, genre_name):
        tx.run(
            "MERGE (g:Genre {name: $name}) "
            "MERGE (b:Book {id: $book_id}) "
            "MERGE (b)-[:BELONGS_TO]->(g)",
            name=genre_name,
            book_id=book_id
        )

    def _create_user(self, tx, user_id):
        tx.run(
            "MERGE (u:User {id: $id})",
            id=user_id
        )

    def _create_review_relation(self, tx, book_name, user_id):
        tx.run(
            "MERGE (b:Book {name: $name}) "
            "MERGE (u:User {id: $user_id}) "
            "MERGE (u)-[:REVIEWED]->(b)",
            name=book_name,
            user_id=user_id
        )
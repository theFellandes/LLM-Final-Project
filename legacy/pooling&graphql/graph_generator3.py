import pandas as pd
import uuid
import logging
import os
from SPARQLWrapper import SPARQLWrapper, JSON
from neo4j import GraphDatabase
from concurrent.futures import ThreadPoolExecutor, as_completed
from docker_reader import DockerComposeReader
from internal.reader.yaml_reader import YAMLReader
# Initialize Neo4j driver
from queue import Queue
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load CSV files
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logging.error(f"Failed to load CSV file {file_path}: {e}")
        raise


class Neo4jHandler:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                max_connection_pool_size=50  # Adjust based on capacity
            )
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    def create_graph(self, books_dir):
        failed_books = Queue()  # Thread-safe queue
        failed_reviews = Queue()

        def write_failed(queue, filename):
            with open(filename, 'w') as file:
                while not queue.empty():
                    file.write(f"{queue.get()}\n")

        with self.driver.session() as session:
            try:
                files = [os.path.join(books_dir, f) for f in os.listdir(books_dir)]
                with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust worker count
                    executor.map(
                        lambda file: self._process_file(file, session, failed_books, failed_reviews),
                        files
                    )
            except Exception as e:
                logging.error(f"Failed to create graph: {e}")
            finally:
                write_failed(failed_books, "failed_books.csv")
                write_failed(failed_reviews, "failed_reviews.csv")

    def _process_file(self, file_path, session, failed_books, failed_reviews):
        try:
            if "book" in file_path:
                df = load_csv(file_path)
                with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust worker count
                    executor.map(
                        lambda row: self._process_book_row(row, session, failed_books),
                        df.iterrows()
                    )
            elif "user_rating" in file_path:
                df = load_csv(file_path)
                with ThreadPoolExecutor(max_workers=5) as executor:
                    executor.map(
                        lambda row: self._process_rating_row(row, session, failed_reviews),
                        df.iterrows()
                    )
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")

    def _process_book_row(self, row, session, failed_books):
        _, row = row
        book_id = str(uuid.uuid4())
        try:
            session.write_transaction(self._create_book, book_id, row)
            authors = row['Authors'].split(",")
            for author in authors:
                session.write_transaction(self._create_author_relation, book_id, author.strip())

            genre = self.fetch_genre(row.get('ISBN'), row.get('Name'))
            if genre:
                for g in genre.split(', '):
                    session.write_transaction(self._create_genre_relation, book_id, g)
            else:
                session.write_transaction(self._create_genre_relation, book_id, "No Genre")
        except Exception as e:
            logging.error(f"Failed to process book row: {e}")
            failed_books.put(row.to_dict())  # Store in thread-safe queue

    def _process_rating_row(self, row, session, failed_reviews):
        _, row = row
        user_id = f"user_{row['ID']}"
        try:
            session.write_transaction(self._create_user, user_id)
            session.write_transaction(self._create_review_relation, row['Name'], user_id)
        except Exception as e:
            logging.error(f"Failed to process rating row: {e}")
            failed_reviews.put(row.to_dict())  # Store in thread-safe queue


    @staticmethod
    def _create_book(tx, book_id, row):
        query = (
            "CREATE (b:Book {"
            "uuid: $book_id, Name: $name, RatingDist1: $rating_dist1, pagesNumber: $pages_number, RatingDist4: $rating_dist4,"
            "RatingDistTotal: $rating_dist_total, PublishMonth: $publish_month, PublishDay: $publish_day, Publisher: $publisher,"
            "CountsOfReview: $counts_of_review, PublishYear: $publish_year, Language: $language, Rating: $rating, RatingDist2: $rating_dist2,"
            "RatingDist5: $rating_dist5, ISBN: $isbn, RatingDist3: $rating_dist3})"
        )
        tx.run(query, book_id=book_id, name=row.get('Name'), rating_dist1=row.get('RatingDist1'),
               pages_number=row.get('pagesNumber'), rating_dist4=row.get('RatingDist4'),
               rating_dist_total=row.get('RatingDistTotal'), publish_month=row.get('PublishMonth'),
               publish_day=row.get('PublishDay'), publisher=row.get('Publisher'),
               counts_of_review=row.get('CountsOfReview'), publish_year=row.get('PublishYear'),
               language=row.get('Language'), rating=row.get('Rating'), rating_dist2=row.get('RatingDist2'),
               rating_dist5=row.get('RatingDist5'), isbn=row.get('ISBN'), rating_dist3=row.get('RatingDist3'))

    @staticmethod
    def _create_author_relation(tx, book_id, author_name):
        query = (
            "MERGE (a:Author {Name: $author_name}) "
            "WITH a "
            "MATCH (b:Book {uuid: $book_id}) "
            "MERGE (b)-[:WRITTEN_BY]->(a)"
        )
        tx.run(query, book_id=book_id, author_name=author_name)

    @staticmethod
    def _create_user(tx, user_id):
        query = "MERGE (u:User {ID: $user_id})"
        tx.run(query, user_id=user_id)

    @staticmethod
    def _create_review_relation(tx, book_name, user_id):
        query = (
            "MATCH (b:Book {Name: $book_name}) "
            "MATCH (u:User {ID: $user_id}) "
            "MERGE (u)-[:REVIEWED_BY]->(b)"
        )
        tx.run(query, book_name=book_name, user_id=user_id)

    @staticmethod
    def _create_genre_relation(tx, book_id, genre_name):
        query = (
            "MERGE (g:Genre {Name: $genre_name}) "
            "WITH g "
            "MATCH (b:Book {uuid: $book_id}) "
            "MERGE (b)-[:BELONGS_TO]->(g)"
        )
        tx.run(query, book_id=book_id, genre_name=genre_name)

    @staticmethod
    def fetch_genre(isbn, name):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)

        query = ("""
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dct: <http://purl.org/dc/terms/>
        SELECT ?genre
        WHERE {
          ?book rdf:type dbo:Book .
          OPTIONAL { ?book dbo:isbn ?isbn FILTER (?isbn = "$isbn") }
          OPTIONAL { ?book rdfs:label ?label FILTER (lang(?label) = "en" && CONTAINS(lcase(?label), lcase("$name"))) }
          OPTIONAL { ?book dct:subject/rdfs:label ?genre FILTER (lang(?genre) = "en") }
        } LIMIT 1
        """).replace("$isbn", isbn or "").replace("$name", name or "")

        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                return result["genre"]["value"]
        except Exception as e:
            logging.error(f"Failed to fetch genre from SPARQL: {e}")
            return None


# Main logic
if __name__ == "__main__":
    books_dir = "../../.cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18"

    try:
        docker_compose_path = "../../docker-compose.yml"
        docker_compose_reader = DockerComposeReader(yaml_file_path=docker_compose_path)
        uri, user, password = docker_compose_reader.get_neo4j_config()

        neo4j_handler = Neo4jHandler(uri, user, password)

        try:
            neo4j_handler.create_graph(books_dir)
        finally:
            neo4j_handler.close()

        logging.info("Graph creation complete!")
    except Exception as e:
        logging.error(f"Script failed: {e}")

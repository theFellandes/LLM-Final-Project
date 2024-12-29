import os
import pandas as pd
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from neo4j import GraphDatabase
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Neo4jHandler:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logging.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def bulk_insert_files_parallel(self, files_dir, prefix, batch_size=100, max_workers=4):
        files = [f for f in os.listdir(files_dir) if f.startswith(prefix)]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_file_with_retries, os.path.join(files_dir, file), prefix, batch_size): file
                for file in files
            }
            for future in as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    future.result()
                    logging.info(f"Successfully processed file: {file_name}")
                except Exception as e:
                    logging.error(f"Failed to process file {file_name}: {e}")

    def _process_file_with_retries(self, file_path, prefix, batch_size, retries=5, delay=2):
        for attempt in range(retries):
            try:
                if prefix == "book":
                    self._process_book_file(file_path, batch_size)
                elif prefix == "user_rating":
                    self._process_user_rating_file(file_path, batch_size)
                return  # Exit loop if successful
            except Exception as e:
                if "Neo.ClientError.Security.AuthenticationRateLimit" in str(e):
                    logging.warning(
                        f"AuthenticationRateLimit error on attempt {attempt + 1} for file {file_path}. Retrying..."
                    )
                    if attempt < retries - 1:
                        sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise e
                else:
                    raise e  # Raise non-recoverable errors immediately

    def _process_book_file(self, file_path, batch_size):
        books_df = pd.read_csv(file_path)

        # Ensure 'uuid' column exists and contains no null values
        if 'uuid' not in books_df.columns:
            logging.warning(f"'uuid' column not found in {file_path}. Generating UUIDs for all rows...")
            books_df['uuid'] = [str(uuid.uuid4()) for _ in range(len(books_df))]
        else:
            books_df['uuid'] = books_df['uuid'].fillna('').apply(
                lambda x: str(uuid.uuid4()) if x == '' else x
            )

        # Handle missing or null pagesNumber
        if 'pagesNumber' not in books_df.columns:
            logging.warning(f"'pagesNumber' column not found in {file_path}. Filling with 0...")
            books_df['pagesNumber'] = 0
        else:
            books_df['pagesNumber'] = books_df['pagesNumber'].fillna(0).astype(int)

        # Replace remaining NaN values in other columns with empty strings
        books_df.fillna('', inplace=True)

        total_books = len(books_df)
        batches = [books_df.iloc[i:i + batch_size] for i in range(0, total_books, batch_size)]

        with self.driver.session() as session:
            for batch in batches:
                session.execute_write(self._insert_books_batch, batch.to_dict('records'))

    def _process_user_rating_file(self, file_path, batch_size):
        ratings_df = pd.read_csv(file_path)
        ratings_df['user_id'] = 'user_' + ratings_df['ID'].astype(str)
        ratings_df.fillna('', inplace=True)

        total_ratings = len(ratings_df)
        batches = [ratings_df.iloc[i:i + batch_size] for i in range(0, total_ratings, batch_size)]

        with self.driver.session() as session:
            for batch in batches:
                session.execute_write(self._insert_user_ratings_batch, batch.to_dict('records'))

    @staticmethod
    def _insert_books_batch(tx, batch):
        query = """
        UNWIND $batch AS row
        MERGE (b:Book {
            uuid: row.book_id,
            Name: row.Name,
            RatingDist1: row.RatingDist1,
            pagesNumber: row.pagesNumber,
            RatingDist4: row.RatingDist4,
            RatingDistTotal: row.RatingDistTotal,
            PublishMonth: row.PublishMonth,
            PublishDay: row.PublishDay,
            Publisher: row.Publisher,
            CountsOfReview: row.CountsOfReview,
            PublishYear: row.PublishYear,
            Language: row.Language,
            Rating: row.Rating,
            RatingDist2: row.RatingDist2,
            RatingDist5: row.RatingDist5,
            ISBN: row.ISBN,
            RatingDist3: row.RatingDist3
        })
        """
        tx.run(query, batch=batch)

    @staticmethod
    def _insert_user_ratings_batch(tx, batch):
        query = """
        UNWIND $batch AS row
        MERGE (u:User {ID: row.user_id})
        MERGE (b:Book {Name: row.Name})
        MERGE (u)-[r:REVIEWED_BY]->(b)
        SET r.rating = row.Rating
        """
        tx.run(query, batch=batch)


# Main Logic
if __name__ == "__main__":
    files_dir = ".cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18"

    try:
        # Get Neo4j configuration
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = "your_password"

        # Initialize Neo4j handler
        neo4j_handler = Neo4jHandler(uri, user, password)

        try:
            # Process book files
            neo4j_handler.bulk_insert_files_parallel(files_dir, prefix="book", batch_size=100, max_workers=4)

            # Process user rating files
            neo4j_handler.bulk_insert_files_parallel(files_dir, prefix="user_rating", batch_size=100, max_workers=4)

        finally:
            neo4j_handler.close()

        logging.info("Parallel processing completed successfully!")
    except Exception as e:
        logging.error(f"Script failed: {e}")

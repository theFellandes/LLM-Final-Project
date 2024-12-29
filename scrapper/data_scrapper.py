import os
import pandas as pd
import kagglehub
import shutil
import logging
from pydantic import BaseModel, Field, field_validator


class Scrapper(BaseModel):
    base_path: str = Field(..., description="The directory path for dataset storage")

    class Config:
        arbitrary_types_allowed = True

    @field_validator("base_path", mode="before")
    def ensure_directory_exists(cls, value):
        """
        Ensure the directory exists. If not, create it.
        """
        if not os.path.exists(value):
            print(f"Directory {value} does not exist. Creating it...")
            os.makedirs(value)
        elif not os.path.isdir(value):
            raise ValueError(f"The path {value} is not a valid directory.")
        return value

    def initialize_dataset(self, already_installed=True):
        """
        Ensures the dataset is available. Downloads the dataset if needed.
        """
        if already_installed:
            print(f"Dataset directory {self.base_path} is ready.")
            return

        if not os.listdir(self.base_path):
            print(f"Directory {self.base_path} is empty. Downloading dataset...")
            self.download_dataset()
        else:
            print(f"Dataset directory {self.base_path} is ready.")

    def download_dataset(self):
        """
        Download the dataset using kagglehub to the specified path.
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        try:
            dataset_path = kagglehub.dataset_download("bahramjannesarr/goodreads-book-datasets-10m")
            # Move downloaded files to the specified base path
            for file_name in os.listdir(dataset_path):
                source = os.path.join(dataset_path, file_name)
                destination = os.path.join(self.base_path, file_name)
                shutil.move(source, destination)
            logging.info(f"Dataset downloaded to {self.base_path}.")
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise

    def read_csv_files(self, chunk_size=10000):
        """
        Generator function to read CSV files and yield book or user rating objects in chunks.
        """
        for file_name in os.listdir(self.base_path):
            file_path = os.path.join(self.base_path, file_name)
            if file_name.startswith("book"):
                print(f"Processing book file: {file_path}")
                yield from self.process_books(file_path, chunk_size)
            elif file_name.startswith("user_rating"):
                print(f"Processing user rating file: {file_path}")
                yield from self.process_user_ratings(file_path, chunk_size)

    def process_books(self, file_path, chunk_size):
        """
        Process book files and yield book objects.
        """
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                yield self.create_book_object(row)

    def process_user_ratings(self, file_path, chunk_size):
        """
        Process user rating files and yield user rating objects.
        """
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                yield self.create_user_rating_object(row)

    @staticmethod
    def create_book_object(row):
        """
        Converts a row into a book object.
        """
        return {
            "id": row.get("Id"),
            "name": row.get("Name"),
            "rating_dist_1": row.get("RatingDist1"),
            "pages_number": row.get("pagesNumber"),
            "rating_dist_4": row.get("RatingDist4"),
            "rating_dist_total": row.get("RatingDistTotal"),
            "publish_month": row.get("PublishMonth"),
            "publish_day": row.get("PublishDay"),
            "publisher": row.get("Publisher"),
            "counts_of_review": row.get("CountsOfReview"),
            "publish_year": row.get("PublishYear"),
            "language": row.get("Language"),
            "authors": row.get("Authors"),
            "rating": row.get("Rating"),
            "rating_dist_2": row.get("RatingDist2"),
            "rating_dist_5": row.get("RatingDist5"),
            "isbn": row.get("ISBN"),
            "rating_dist_3": row.get("RatingDist3")
        }

    @staticmethod
    def create_user_rating_object(row):
        """
        Converts a row into a user rating object.
        """
        return {
            "id": row.get("ID"),
            "name": row.get("Name"),
            "rating": row.get("Rating")
        }


# Example Usage
if __name__ == "__main__":
    # Provide a custom path
    custom_path = os.path.abspath(".cache/kagglehub/datasets/bahramjannesarr/goodreads-book-datasets-10m/versions/18")
    scrapper = Scrapper(base_path=custom_path)

    # Explicitly initialize the dataset (download if necessary)
    scrapper.initialize_dataset(already_installed=True)

    # Process books and user ratings
    for obj in scrapper.read_csv_files(chunk_size=10000):
        print(obj)

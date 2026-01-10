import os
import sys
import pickle
import pandas as pd
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.exception.exception_handler import AppException


class DataTransformation:
    def __init__(self, app_config=AppConfiguration()):
        try:
            # Load configuration objects
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_transformer(self):
        try:
            # Load the cleaned CSV data safely
            df = pd.read_csv(
                self.data_transformation_config.clean_data_file_path,
                sep=',',  # adjust if your CSV uses ';'
                encoding='latin-1',
                low_memory=False  # avoids dtype warnings
            )

            # Create pivot table: rows = book titles, columns = users, values = ratings
            book_pivot = df.pivot_table(
                index='title',
                columns='user_id',
                values='rating',
                fill_value=0  # replaces NaN with 0
            )
            logging.info(f"Shape of book pivot table: {book_pivot.shape}")

            # Ensure directories exist
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)

            # Save pivot table
            pivot_file_path = os.path.join(
                self.data_transformation_config.transformed_data_dir, "transformed_data.pkl"
            )
            pickle.dump(book_pivot, open(pivot_file_path, 'wb'))
            logging.info(f"Saved pivot table to {pivot_file_path}")

            # Save book names
            book_names = book_pivot.index
            book_names_file = os.path.join(
                self.data_validation_config.serialized_objects_dir, "book_names.pkl"
            )
            pickle.dump(book_names, open(book_names_file, 'wb'))
            logging.info(f"Saved book names to {book_names_file}")

            # Save full pivot table for web app
            book_pivot_file = os.path.join(
                self.data_validation_config.serialized_objects_dir, "book_pivot.pkl"
            )
            pickle.dump(book_pivot, open(book_pivot_file, 'wb'))
            logging.info(f"Saved book pivot object to {book_pivot_file}")

        except Exception as e:
            raise AppException(e, sys) from e

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20} Data Transformation log started {'='*20}")
            self.get_data_transformer()
            logging.info(f"{'='*20} Data Transformation log completed {'='*20}\n\n")
        except Exception as e:
            raise AppException(e, sys) from e


# Example of proper usage
if __name__ == "__main__":
    try:
        transformer = DataTransformation()
        transformer.initiate_data_transformation()
    except Exception as e:
        print(f"Data transformation failed: {e}")

# data ingestion with S3 upload (Class + @staticmethod)
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections import s3_connection


class DataIngestion:
    @staticmethod
    def load_params(params_path: str) -> dict:
        """Load parameters from a YAML file."""
        try:
            with open(params_path, 'r') as file:
                params = yaml.safe_load(file)
            logging.debug('Parameters retrieved from %s', params_path)
            return params
        except FileNotFoundError:
            logging.error('File not found: %s', params_path)
            raise
        except yaml.YAMLError as e:
            logging.error('YAML error: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error: %s', e)
            raise

    @staticmethod
    def load_data(data_url: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(data_url)
            logging.info('Data loaded from %s', data_url)
            return df
        except pd.errors.ParserError as e:
            logging.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the data: %s', e)
            raise

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data."""
        try:
            logging.info("pre-processing...")
            final_df = df[df['sentiment'].isin(['positive', 'negative'])]
            final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
            logging.info('Data preprocessing completed')
            return final_df
        except KeyError as e:
            logging.error('Missing column in the dataframe: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error during preprocessing: %s', e)
            raise

    @staticmethod
    def save_data_to_s3(train_data: pd.DataFrame, test_data: pd.DataFrame, s3_client, bucket_name: str, prefix: str) -> None:
        """
        Save the train and test datasets directly to S3.
        """
        try:
            # Convert DataFrames to CSV strings
            train_csv = train_data.to_csv(index=False)
            test_csv = test_data.to_csv(index=False)

            # Upload to S3 (using s3_operations)
            s3_client.upload_file_from_string(train_csv, bucket_name, f"{prefix}/train.csv")
            s3_client.upload_file_from_string(test_csv, bucket_name, f"{prefix}/test.csv")

            logging.debug('Train and test data uploaded to S3 at prefix %s', prefix)
        except Exception as e:
            logging.error('Unexpected error occurred while uploading the data to S3: %s', e)
            raise

    @staticmethod
    def save_data_to_s3():
        try:
            params = DataIngestion.load_params(params_path='params.yaml')
            test_size = params['data_ingestion']['test_size']
            bucket_name = params['data_ingestion']['bucket_name']
            prefix = params['data_ingestion'].get('prefix', 'raw')

            # Load raw data from GitHub (or S3 if needed)
            df = DataIngestion.load_data(
                data_url='https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/data.csv'
            )

            # Initialize S3 client
            s3 = s3_connection.s3_operations(
                bucket_name=bucket_name,
                access_key=params['aws']['access_key'],
                secret_key=params['aws']['secret_key']
            )

            final_df = DataIngestion.preprocess_data(df)
            train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

            # Save to S3 instead of local disk
            DataIngestion.save_data_to_s3(train_data, test_data, s3, bucket_name, prefix)

        except Exception as e:
            logging.error('Failed to complete the data ingestion process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    DataIngestion.save_data_to_s3()

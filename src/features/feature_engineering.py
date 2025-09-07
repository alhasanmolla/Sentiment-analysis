# feature engineering
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import yaml
from src.logger import logging
import pickle


class FeatureEngineer:

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
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from a CSV file and fill NaN with empty string."""
        try:
            df = pd.read_csv(file_path)
            df.fillna('', inplace=True)
            logging.info('Data loaded and NaNs filled from %s', file_path)
            return df
        except pd.errors.ParserError as e:
            logging.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the data: %s', e)
            raise

    @staticmethod
    def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
        """Apply Bag of Words (CountVectorizer) to the dataset."""
        try:
            logging.info("Applying Bag of Words transformation...")
            vectorizer = CountVectorizer(max_features=max_features)
            # vectorizer = TfidfVectorizer(max_features=max_features)

            # Separate features and labels
            X_train = train_data['review'].values
            y_train = train_data['sentiment'].values
            X_test = test_data['review'].values
            y_test = test_data['sentiment'].values

            # Transform using BOW
            X_train_bow = vectorizer.fit_transform(X_train)
            X_test_bow = vectorizer.transform(X_test)

            # Convert back to DataFrames
            train_df = pd.DataFrame(X_train_bow.toarray())
            train_df['label'] = y_train

            test_df = pd.DataFrame(X_test_bow.toarray())
            test_df['label'] = y_test

            # Save the vectorizer
            os.makedirs("models", exist_ok=True)
            pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

            logging.info('Bag of Words applied and data transformed')
            return train_df, test_df
        except Exception as e:
            logging.error('Error during Bag of Words transformation: %s', e)
            raise

    @staticmethod
    def save_data(df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to a CSV file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            logging.info('Data saved to %s', file_path)
        except Exception as e:
            logging.error('Failed to save data: %s', e)
            raise

    @staticmethod
    def feature_eng():
        """Run the complete feature engineering pipeline."""
        try:
            # Load parameters
            params = FeatureEngineer.load_params('params.yaml')
            max_features = params['feature_engineering']['max_features']

            # Load processed data 
            train_data = FeatureEngineer.load_data('./data/interim/train_processed.csv')
            test_data = FeatureEngineer.load_data('./data/interim/test_processed.csv')

            # Apply Bag of Words
            train_df, test_df = FeatureEngineer.apply_bow(train_data, test_data, max_features)

            # return train_df, test_df

            # Save engineered features
            FeatureEngineer.save_data(train_df, os.path.join("./data", "processed", "train_bow.csv"))
            FeatureEngineer.save_data(test_df, os.path.join("./data", "processed", "test_bow.csv"))

            logging.info("Feature engineering completed successfully")

        except Exception as e:
            logging.error('Failed to complete the feature engineering process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    FeatureEngineer.feature_eng()

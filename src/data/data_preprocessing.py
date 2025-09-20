# data preprocessing

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging

# Download required NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')


class DataPreprocessing:

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess a single text string:
        - Remove URLs, numbers, punctuation
        - Convert to lowercase
        - Remove stopwords
        - Lemmatize words

        Args:
            text (str): Input text

        Returns:
            str: Cleaned and preprocessed text
        """
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("english"))

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove numbers
        text = ''.join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")  # Special char remove
        text = re.sub('\s+', ' ', text).strip()
        # Remove stopwords
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

        return text

    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame, col: str = 'text') -> pd.DataFrame:
        """
        Preprocess a DataFrame by applying text preprocessing to a specific column.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.
            col (str): The name of the column containing text.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        df[col] = df[col].apply(DataPreprocessing.preprocess_text)

        # Drop rows with NaN values
        df = df.dropna(subset=[col])

        logging.info("Data pre-processing completed")
        return df

    @staticmethod
    def preprocessing():
        """
        Orchestrates the preprocessing:
        - Loads train/test data
        - Preprocesses text column
        - Saves the processed data to ./data/interim
        """
        try:
            # Load raw data
            train_data = pd.read_csv('./data/raw/train.csv')
            test_data = pd.read_csv('./data/raw/test.csv')
            logging.info('Raw data loaded properly')

            # Preprocess the review column
            train_processed_data = DataPreprocessing.preprocess_dataframe(train_data, 'review')
            # print(train_processed_data)
            test_processed_data = DataPreprocessing.preprocess_dataframe(test_data, 'review')
            # print(test_processed_data)
            logging.info('Text data pre-processed successfully')

            # return train_processed_data, test_processed_data

            # Save processed data
            data_path = os.path.join("./data", "interim")
            os.makedirs(data_path, exist_ok=True)

            train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
            test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

            logging.info('Processed data saved to %s', data_path)
        except Exception as e:
            logging.error('Failed to complete the data transformation process: %s', e)
            print(f"Error: {e}")

        return{"train_processed_data": train_processed_data, "test_processed_data": test_processed_data}


if __name__ == '__main__':
    DataPreprocessing.preprocessing()

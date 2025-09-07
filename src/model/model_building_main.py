import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import yaml
from src.logger import logging


class ModelBuilding:
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            logging.info('Data loaded from %s', file_path)
            return df
        except pd.errors.ParserError as e:
            logging.error('Failed to parse the CSV file: %s', e)
            raise
        except Exception as e:
            logging.error('Unexpected error occurred while loading the data: %s', e)
            raise

    @staticmethod
    def train_and_tune_models(X_train: np.ndarray, y_train: np.ndarray):
        """Train multiple models with hyperparameter tuning and return the best one."""
        try:
            models = {
                "LogisticRegression": (LogisticRegression(solver="liblinear"), {
                    "C": [0.01, 0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"]
                }),
                "NaiveBayes": (MultinomialNB(), {
                    "alpha": [0.01, 0.1, 0.5, 1.0, 5.0]
                }),
                "LinearSVC": (LinearSVC(), {
                    "C": [0.01, 0.1, 1, 10, 100]
                })
            }

            best_model = None
            best_score = 0
            best_name = ""

            for name, (model, params) in models.items():
                logging.info(f"Training and tuning {name}...")
                grid = GridSearchCV(model, params, cv=3, scoring="accuracy", n_jobs=-1)
                grid.fit(X_train, y_train)

                logging.info(f"{name} best score: {grid.best_score_}, params: {grid.best_params_}")

                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    best_model = grid.best_estimator_
                    best_name = name

            logging.info(f"Best Model: {best_name} with accuracy {best_score}")
            return best_model

        except Exception as e:
            logging.error("Error during model training and tuning: %s", e)
            raise

    @staticmethod
    def save_model(model, file_path: str) -> None:
        """Save the trained model to a file."""
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
            logging.info('Best model saved to %s', file_path)
        except Exception as e:
            logging.error('Error occurred while saving the model: %s', e)
            raise

    @staticmethod
    def best_model():
        try:
            # Load training data
            train_data = ModelBuilding.load_data('./data/processed/train_bow.csv')
            X_train = train_data.iloc[:, :-1].values
            y_train = train_data.iloc[:, -1].values

            # Train and select the best model
            best_clf = ModelBuilding.train_and_tune_models(X_train, y_train)

            # Save the best model
            ModelBuilding.save_model(best_clf, 'models/model.pkl')

        except Exception as e:
            logging.error('Failed to complete the model building process: %s', e)
            print(f"Error: {e}")


if __name__ == '__main__':
    ModelBuilding.best_model()

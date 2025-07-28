"""
Language Detection System
A comprehensive machine learning solution for automated language identification.
Author: Awais Mughal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from typing import Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LanguageDetectionSystem:
    """
    A comprehensive language detection system using multiple ML algorithms.

    This system implements and compares various machine learning approaches
    for language identification, providing both word-level and character-level
    feature extraction capabilities.
    """

    def __init__(self, data_file: str = "language.csv"):
        """
        Initialize the Language Detection System.

        Args:
            data_file (str): Path to the CSV file containing training data
        """
        self.data_file = data_file
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_accuracy = 0.0
        self.feature_names = None

    def load_data(self) -> pd.DataFrame:
        """
        Load training data from CSV file.

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If the data file is not found
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_csv(self.data_file)
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")

            # Validate required columns
            required_columns = ['text', 'language']
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            return df

        except FileNotFoundError:
            logger.error(f"Data file '{self.data_file}' not found")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_text(self, text: str) -> str:
        """
        Apply text preprocessing steps.

        Args:
            text (str): Raw text input

        Returns:
            str: Preprocessed text
        """
        if pd.isna(text):
            return ""

        # Convert to lowercase and normalize whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def analyze_dataset(self, df: pd.DataFrame) -> None:
        """
        Analyze and display dataset characteristics.

        Args:
            df (pd.DataFrame): Dataset to analyze
        """
        logger.info("Dataset Analysis:")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Language distribution
        language_dist = df['language'].value_counts()
        logger.info(f"Language distribution:\n{language_dist}")

        # Text length statistics
        text_lengths = df['text'].str.len()
        logger.info(f"Text length statistics:")
        logger.info(f"Mean: {text_lengths.mean():.2f}")
        logger.info(f"Median: {text_lengths.median():.2f}")
        logger.info(f"Min: {text_lengths.min()}")
        logger.info(f"Max: {text_lengths.max()}")

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by preprocessing and splitting.

        Args:
            df (pd.DataFrame): Raw dataset

        Returns:
            Tuple containing train/test splits for features and labels
        """
        logger.info("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        logger.info(f"Samples after cleaning: {len(df)}")

        # Prepare features and labels
        X = df['processed_text'].values
        y = df['language'].values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Testing samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def initialize_models(self) -> Dict[str, Pipeline]:
        """
        Initialize machine learning models with different feature extraction approaches.

        Returns:
            Dict[str, Pipeline]: Dictionary of model pipelines
        """
        models = {
            'Naive Bayes + TF-IDF': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
                ('classifier', MultinomialNB())
            ]),

            'Logistic Regression + TF-IDF': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),

            'SVM + TF-IDF': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
                ('classifier', SVC(kernel='linear', random_state=42, probability=True))
            ]),

            'Character-level TF-IDF + Logistic Regression': Pipeline([
                ('char_tfidf', TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(2, 5),
                    max_features=10000
                )),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ])
        }

        return models

    def train_and_evaluate_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                  y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Train and evaluate all models.

        Args:
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing labels

        Returns:
            Dict[str, float]: Model performance results
        """
        models = self.initialize_models()
        results = {}

        logger.info("Training and evaluating models...")
        logger.info("-" * 80)

        for name, model in models.items():
            logger.info(f"Training {name}...")

            try:
                # Train model
                model.fit(X_train, y_train)

                # Evaluate model
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                results[name] = accuracy
                self.models[name] = model

                logger.info(f"{name} - Accuracy: {accuracy:.4f}")

                # Track best model
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = model
                    self.best_model_name = name

            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = 0.0

        logger.info("-" * 80)
        logger.info(f"Best model: {self.best_model_name} (Accuracy: {self.best_accuracy:.4f})")

        return results

    def detailed_evaluation(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Perform detailed evaluation of the best model.

        Args:
            X_test: Testing features
            y_test: Testing labels
        """
        if not self.best_model:
            logger.warning("No best model available for detailed evaluation")
            return

        logger.info(f"Detailed evaluation of {self.best_model_name}:")

        # Predictions
        y_pred = self.best_model.predict(X_test)

        # Classification report
        logger.info("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion matrix visualization
        self._plot_confusion_matrix(y_test, y_pred)

        # Feature analysis for character-level models
        if 'Character-level' in self.best_model_name:
            self._analyze_character_features()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Plot confusion matrix for model evaluation.

        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.best_model.classes_,
                    yticklabels=self.best_model.classes_)
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def _analyze_character_features(self) -> None:
        """
        Analyze character-level features for interpretability.
        """
        try:
            vectorizer = self.best_model.named_steps['char_tfidf']
            classifier = self.best_model.named_steps['classifier']
            feature_names = vectorizer.get_feature_names_out()

            logger.info("Top character n-grams by language:")

            for i, language in enumerate(classifier.classes_):
                top_indices = np.argsort(classifier.coef_[i])[-15:]
                top_features = [feature_names[idx] for idx in top_indices]
                logger.info(f"{language.upper()}: {', '.join(top_features)}")

        except Exception as e:
            logger.error(f"Error analyzing character features: {str(e)}")

    def test_sample_predictions(self) -> None:
        """
        Test the model with sample texts to demonstrate functionality.
        """
        if not self.best_model:
            logger.warning("No trained model available for testing")
            return

        sample_texts = [
            "Bonjour, comment allez-vous aujourd'hui?",
            "Hello, how are you doing today?",
            "Je suis très heureux de vous rencontrer.",
            "I am very happy to meet you.",
            "C'est une belle journée pour une promenade.",
            "It's a beautiful day for a walk.",
            "Gracias por tu ayuda, eres muy amable.",
            "Danke für deine Hilfe, du bist sehr freundlich."
        ]

        logger.info("Testing with sample texts:")
        logger.info("-" * 60)

        for text in sample_texts:
            prediction = self.best_model.predict([text])[0]
            probabilities = self.best_model.predict_proba([text])[0]
            confidence = max(probabilities)

            logger.info(f"Text: '{text}'")
            logger.info(f"Predicted: {prediction} (Confidence: {confidence:.3f})")
            logger.info("-" * 60)

    def save_model(self, filepath: str = 'language_detection_model.pkl') -> None:
        """
        Save the best trained model to disk.

        Args:
            filepath (str): Path to save the model
        """
        if not self.best_model:
            logger.warning("No trained model to save")
            return

        try:
            joblib.dump(self.best_model, filepath)
            logger.info(f"Model saved successfully to '{filepath}'")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, filepath: str = 'language_detection_model.pkl') -> None:
        """
        Load a pre-trained model from disk.

        Args:
            filepath (str): Path to the saved model
        """
        try:
            self.best_model = joblib.load(filepath)
            logger.info(f"Model loaded successfully from '{filepath}'")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Make a prediction on new text.

        Args:
            text (str): Text to classify

        Returns:
            Tuple[str, float]: Predicted language and confidence score
        """
        if not self.best_model:
            raise ValueError("No trained model available. Train a model first.")

        preprocessed_text = self.preprocess_text(text)
        prediction = self.best_model.predict([preprocessed_text])[0]
        probabilities = self.best_model.predict_proba([preprocessed_text])[0]
        confidence = max(probabilities)

        return prediction, confidence

    def run_complete_pipeline(self) -> Dict[str, float]:
        """
        Execute the complete language detection pipeline.

        Returns:
            Dict[str, float]: Model performance results
        """
        try:
            # Load and analyze data
            df = self.load_data()
            self.analyze_dataset(df)

            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df)

            # Train and evaluate models
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)

            # Detailed evaluation
            self.detailed_evaluation(X_test, y_test)

            # Test with samples
            self.test_sample_predictions()

            # Save best model
            self.save_model()

            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise


def main():
    """
    Main execution function.
    """
    # Initialize the language detection system
    detector = LanguageDetectionSystem("language.csv")

    # Run the complete pipeline
    try:
        results = detector.run_complete_pipeline()

        # Display final results
        logger.info("Final Model Performance Summary:")
        logger.info("=" * 80)
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{model_name:<40} | Accuracy: {accuracy:.4f}")
        logger.info("=" * 80)

        # Performance assessment
        if detector.best_accuracy >= 0.99:
            logger.info("SUCCESS: Achieved target accuracy of 99%")
        else:
            improvement_needed = 0.99 - detector.best_accuracy
            logger.info(f"TARGET MISSED: Need {improvement_needed:.4f} more accuracy points")

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        return 1

    return 0


def predict_language(text: str, model_path: str = 'language_detection_model.pkl') -> Tuple[str, float]:
    """
    Utility function to load model and make predictions.

    Args:
        text (str): Text to classify
        model_path (str): Path to saved model

    Returns:
        Tuple[str, float]: Predicted language and confidence
    """
    detector = LanguageDetectionSystem()
    detector.load_model(model_path)
    return detector.predict(text)


if __name__ == "__main__":
    exit(main())

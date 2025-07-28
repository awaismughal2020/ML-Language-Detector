import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load and preprocess the data
def load_data_from_google_drive(file_id):
    """
    Load data from Google Drive using the file ID
    """
    df = pd.read_csv(file_id)
    return df


def preprocess_text(text):
    """
    Basic text preprocessing
    """
    if pd.isna(text):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optional: Remove punctuation (comment out if you want to keep it)
    # text = text.translate(str.maketrans('', '', string.punctuation))

    return text


def create_language_detection_model(file_id):
    """
    Main function to create and train the language detection model
    """

    # Load the data
    print("Loading data from CSV...")
    df = load_data_from_google_drive(file_id)

    # Display basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())

    # Assuming the dataset has columns 'text' and 'language'
    # Adjust column names based on your actual dataset
    if 'text' not in df.columns or 'language' not in df.columns:
        print("Expected columns: 'text' and 'language'")
        print("Please verify your dataset structure and adjust column names if needed")
        return None

    # Check language distribution
    print("\nLanguage distribution:")
    print(df['language'].value_counts())

    # Preprocess the text
    print("Preprocessing text...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # Remove empty texts
    df = df[df['processed_text'].str.len() > 0]

    # Split the dataset (80/20)
    X = df['processed_text']
    y = df['language']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Define multiple models to try
    models = {
        # 'Naive Bayes + TF-IDF': Pipeline([
        #     ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
        #     ('nb', MultinomialNB())
        # ]),
        'Logistic Regression + TF-IDF': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        # 'SVM + TF-IDF': Pipeline([
        #     ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=10000)),
        #     ('svm', SVC(kernel='linear', random_state=42))
        # ]),
        # 'Character-level TF-IDF + Logistic Regression': Pipeline([
        #     ('char_tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=10000)),
        #     ('lr', LogisticRegression(random_state=42, max_iter=1000))
        # ])
    }

    # Train and evaluate models
    results = {}
    best_model = None
    best_accuracy = 0

    print("\nTraining and evaluating models...")
    print("-" * 60)

    for name, model in models.items():
        print(f"Training {name}...")

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy

        print(f"{name} - Accuracy: {accuracy:.4f}")

        # Keep track of the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = (name, model)

    print("-" * 60)
    print(f"Best model: {best_model[0]} with accuracy: {best_accuracy:.4f}")

    # Detailed evaluation of the best model
    if best_model:
        print(f"\nDetailed evaluation of {best_model[0]}:")
        y_pred_best = best_model[1].predict(X_test)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_best))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=best_model[1].classes_,
                    yticklabels=best_model[1].classes_)
        plt.title(f'Confusion Matrix - {best_model[0]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        # Feature importance for character-level model
        if 'Character-level' in best_model[0]:
            vectorizer = best_model[1].named_steps['char_tfidf']
            classifier = best_model[1].named_steps['lr']

            feature_names = vectorizer.get_feature_names_out()

            # Get top features for each language
            for i, language in enumerate(classifier.classes_):
                top_indices = np.argsort(classifier.coef_[i])[-20:]
                top_features = [feature_names[idx] for idx in top_indices]
                print(f"\nTop character n-grams for {language}:")
                print(top_features)

    # Test with sample texts
    print("\nTesting with sample texts:")
    sample_texts = [
        "Bonjour, comment allez-vous aujourd'hui?",
        "Hello, how are you doing today?",
        "Je suis très heureux de vous rencontrer.",
        "I am very happy to meet you.",
        "C'est une belle journée pour une promenade.",
        "It's a beautiful day for a walk."
    ]

    for text in sample_texts:
        prediction = best_model[1].predict([text])[0]
        probability = max(best_model[1].predict_proba([text])[0])
        print(f"Text: '{text}' -> Predicted: {prediction} (confidence: {probability:.3f})")

    return best_model[1], results


# Usage example
if __name__ == "__main__":
    # Extract the file ID from the Google Drive URL
    # https://drive.google.com/file/d/13P7fjttl_dKmJmUvAXjigggalrocecd/edit?usp=sharing
    file_id = "language.csv"

    # Create and train the model
    trained_model, model_results = create_language_detection_model(file_id)

    # Save the best model (optional)
    if trained_model:
        import joblib

        joblib.dump(trained_model, 'language_detection_model.pkl')
        print("\nModel saved as 'language_detection_model.pkl'")



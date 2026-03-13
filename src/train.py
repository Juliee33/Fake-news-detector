"""
train.py

Fake News Detection – Training & Evaluation Pipeline

This script implements a structured experimental framework for
text-based fake news detection using supervised machine learning.

Pipeline Stages:
1. Load cleaned dataset
2. Perform train-test split (80/20)
3. Apply TF-IDF feature extraction
4. Train multiple baseline classifiers
5. Evaluate using:
   - Confusion matrices
   - Accuracy, Precision, Recall, F1-score
   - 5-fold cross-validation
6. Perform structured error analysis
7. Save models, vectorizer, and results for reproducibility

This modular and documented structure ensures transparency,
reproducibility, and research-grade evaluation.
"""

import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ---------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------

DATA_PATH = "../data/processed/cleaned_data.csv"
MODELS_PATH = "../models/"
RESULTS_PATH = "../results/"

os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)


# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------

def load_data():
    """
    Loads the cleaned dataset used for model training.

    Returns:
        pd.DataFrame: Dataset containing 'text' and 'label'.
    """
    print("Loading cleaned dataset...")
    data = pd.read_csv(DATA_PATH)
    return data


# ---------------------------------------------------------
# Training and Evaluation Pipeline
# ---------------------------------------------------------

def train_models():
    """
    Executes the full supervised learning pipeline:
    - Data splitting
    - Feature extraction
    - Model training
    - Quantitative evaluation
    - Cross-validation
    - Error analysis
    - Model persistence
    """

    # Load dataset
    data = load_data()

    print("\nClass Distribution:")
    print(data["label"].value_counts())

    X = data["text"]
    y = data["label"]

    # -----------------------------------------------------
    # Train-Test Split (Prevents Data Leakage)
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # -----------------------------------------------------
    # TF-IDF Feature Engineering
    # -----------------------------------------------------

    print("\nVectorising text using TF-IDF...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7,
        max_features=5000
    )

    # Fit only on training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Separate full transformation for cross-validation
    X_full_tfidf = vectorizer.transform(X)

    # -----------------------------------------------------
    # Model Definitions
    # -----------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB()
    }

    results = []

    # -----------------------------------------------------
    # Model Training Loop
    # -----------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining {name}...")

        # Train model
        model.fit(X_train_tfidf, y_train)

        # Predict on test set
        predictions = model.predict(X_test_tfidf)

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        cm = confusion_matrix(y_test, predictions)

        print(f"\nConfusion Matrix for {name}:")
        print(cm)

        # -------------------------------------------------
        # Quantitative Metrics
        # -------------------------------------------------

        acc = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # -------------------------------------------------
        # Cross-Validation (5-Fold)
        # -------------------------------------------------

        cv_scores = cross_val_score(
            model,
            X_full_tfidf,
            y,
            cv=5,
            scoring="f1"
        )

        print(f"\n{name} 5-Fold CV F1 Scores: {cv_scores}")
        print(f"{name} Mean CV F1: {cv_scores.mean():.4f}")
        print(f"{name} Std Dev: {cv_scores.std():.4f}")

        # -------------------------------------------------
        # Structured Error Analysis (Qualitative Insight)
        # -------------------------------------------------

        if name in ["Linear SVM", "Naive Bayes"]:

            false_positives = X_test[(predictions == 1) & (y_test == 0)]
            false_negatives = X_test[(predictions == 0) & (y_test == 1)]

            error_df = pd.DataFrame({
                "Text": pd.concat([
                    false_positives.head(5),
                    false_negatives.head(5)
                ]),
                "Error Type": (
                    ["False Positive"] * min(5, len(false_positives)) +
                    ["False Negative"] * min(5, len(false_negatives))
                )
            })

            error_filename = name.replace(" ", "_").lower() + "_errors.csv"
            error_df.to_csv(
                os.path.join(RESULTS_PATH, error_filename),
                index=False
            )

            print(f"\nSaved sample misclassifications for {name} to {error_filename}")

        # Save trained model
        model_filename = name.replace(" ", "_").lower() + ".pkl"
        with open(os.path.join(MODELS_PATH, model_filename), "wb") as f:
            pickle.dump(model, f)

    # -----------------------------------------------------
    # Save Vectorizer
    # -----------------------------------------------------

    with open(os.path.join(MODELS_PATH, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    # -----------------------------------------------------
    # Save Evaluation Results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "metrics.csv"), index=False)

    print("\nTraining complete. Results saved.")
    print("\nFinal Test Set Metrics:")
    print(results_df)


# ---------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    train_models()
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, fbeta_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import prepare_text

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def choose_threshold(y_true, probabilities):
    """Choose a threshold on validation data using F2 for the fraud class."""
    best_threshold, best_f2 = 0.50, -1.0
    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=2, zero_division=0)
        if score > best_f2:
            best_threshold, best_f2 = float(threshold), float(score)
    return best_threshold, float(best_f2)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put fake_job_postings.csv inside the data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    if "fraudulent" not in df.columns:
        raise ValueError("Expected target column 'fraudulent' in the CSV.")

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["fraudulent"]).copy()

    df["fraudulent"] = (
        df["fraudulent"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"f": "0", "t": "1"})
    )

    df["fraudulent"] = pd.to_numeric(
        df["fraudulent"], errors="raise"
    ).astype(int)
    if not set(df["fraudulent"].unique()).issubset({0, 1}):
        raise ValueError("The 'fraudulent' target must contain only 0 and 1.")

    X = prepare_text(df)
    y = df["fraudulent"]

    # 70% train, 15% validation, 15% test; keep class proportions similar.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=100_000,
            min_df=2, sublinear_tf=True, strip_accents="unicode"
        )),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1500, random_state=RANDOM_STATE
        )),
    ])

    print("Training Logistic Regression + TF-IDF...")
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    threshold, validation_f2 = choose_threshold(y_val, val_probabilities)

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)

    metrics = {
        "threshold_selected_on_validation": threshold,
        "validation_f2_fraud_class": validation_f2,
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "test_precision_fraud_class": float(precision_score(y_test, test_predictions, zero_division=0)),
        "test_recall_fraud_class": float(recall_score(y_test, test_predictions, zero_division=0)),
        "test_f1_fraud_class": float(f1_score(y_test, test_predictions, zero_division=0)),
        "test_f2_fraud_class": float(fbeta_score(y_test, test_predictions, beta=2, zero_division=0)),
        "test_confusion_matrix": confusion_matrix(y_test, test_predictions).tolist(),
        "test_classification_report": classification_report(
            y_test, test_predictions, output_dict=True, zero_division=0
        ),
        "test_rows": int(len(y_test)),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "jobguard_model.joblib")
    with open(MODEL_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "actual_fraudulent": y_test.to_numpy(),
        "model_score": test_probabilities,
        "predicted_fraudulent": test_predictions,
    }).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print("\nTest results (fraudulent class):")
    print(f"Threshold: {threshold:.2f}")
    print(f"Precision: {metrics['test_precision_fraud_class']:.3f}")
    print(f"Recall:    {metrics['test_recall_fraud_class']:.3f}")
    print(f"F1:        {metrics['test_f1_fraud_class']:.3f}")
    print(f"F2:        {metrics['test_f2_fraud_class']:.3f}")
    print("Confusion matrix [ [TN, FP], [FN, TP] ]:")
    print(metrics["test_confusion_matrix"])
    print("\nSaved model and evaluation outputs.")


if __name__ == "__main__":
    main()

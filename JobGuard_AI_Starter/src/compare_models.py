
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.preprocessing import prepare_text
from src.train import choose_threshold, RANDOM_STATE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fake_job_postings.csv"
OUTPUT = ROOT / "outputs"

# Load and clean data using existing training logic.
df = pd.read_csv(DATA).drop_duplicates()
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

X = prepare_text(df)
y = df["fraudulent"]

# Reproduce the original 70/15/15 split.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=y_temp
)

# Load existing Logistic Regression model.
lr = joblib.load(ROOT / "models" / "jobguard_model.joblib")

with open(ROOT / "models" / "threshold.json") as f:
    lr_threshold = json.load(f)["threshold"]

# Train Naive Bayes baseline.
nb = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=100_000,
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode"
    )),
    ("classifier", MultinomialNB())
])

nb.fit(X_train, y_train)

# Tune baseline threshold on validation only.
nb_val_prob = nb.predict_proba(X_val)[:, 1]
nb_threshold, _ = choose_threshold(y_val, nb_val_prob)

models = {
    "Logistic Regression": (lr, lr_threshold),
    "Multinomial Naive Bayes": (nb, nb_threshold),
}

results = []
OUTPUT.mkdir(exist_ok=True)

for name, (model, threshold) in models.items():
    probability = model.predict_proba(X_test)[:, 1]
    pred = (probability >= threshold).astype(int)

    results.append({
        "Model": name,
        "Threshold": round(threshold, 4),
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "F2": fbeta_score(y_test, pred, beta=2, zero_division=0)
    })

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Genuine", "Fraud"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(name)
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT / f"{filename}_confusion_matrix.png")
    plt.close()

comparison = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(comparison.round(4).to_string(index=False))

comparison.to_csv(OUTPUT / "model_comparison.csv", index=False)
print("\nResults saved to outputs/")


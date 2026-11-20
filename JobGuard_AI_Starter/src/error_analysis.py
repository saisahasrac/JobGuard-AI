from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fake_job_postings.csv"
PREDICTIONS = ROOT / "outputs" / "test_predictions.csv"
OUTPUT = ROOT / "outputs" / "detailed_error_analysis.csv"

df = pd.read_csv(DATA).drop_duplicates().copy()
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

train_df, temp_df = train_test_split(
    df, test_size=0.30,
    random_state=42,
    stratify=df["fraudulent"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50,
    random_state=42,
    stratify=temp_df["fraudulent"]
)

predictions = pd.read_csv(PREDICTIONS)

assert len(test_df) == len(predictions), "Test split size mismatch"
assert (
    test_df["fraudulent"].to_numpy()
    == predictions["actual_fraudulent"].to_numpy()
).all(), "Test labels do not match saved predictions"

details = test_df.reset_index(drop=True).copy()

details["model_score"] = predictions["model_score"].to_numpy()
details["predicted_fraudulent"] = predictions[
    "predicted_fraudulent"
].to_numpy()

errors = details[
    details["fraudulent"] != details["predicted_fraudulent"]
].copy()

errors["error_type"] = errors["fraudulent"].map({
    0: "False Positive",
    1: "False Negative"
})

errors["distance_from_threshold"] = abs(
    errors["model_score"] - 0.53
)

columns = [
    "title",
    "description",
    "company_profile",
    "requirements",
    "fraudulent",
    "predicted_fraudulent",
    "model_score",
    "error_type",
    "distance_from_threshold"
]

errors = errors.reindex(columns=columns)
errors.to_csv(OUTPUT, index=False)

print("Total errors:", len(errors))
print(errors["error_type"].value_counts())
print("\nDetailed error analysis saved to:", OUTPUT)

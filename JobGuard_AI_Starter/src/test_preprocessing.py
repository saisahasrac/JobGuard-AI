# ============================================================
# JOBGUARD-AI
# PREPROCESSING TEST
# ============================================================

import pandas as pd
from pathlib import Path

from preprocessing import (
    preprocess_data,
    validate_preprocessed_data,
    preprocessing_summary
)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "fake_job_postings.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# LOAD ORIGINAL DATA
# ------------------------------------------------------------

print("=" * 60)
print("JOBGUARD-AI PREPROCESSING TEST")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nOriginal Dataset Loaded")
print("Shape:", df.shape)

# ------------------------------------------------------------
# RUN PREPROCESSING
# ------------------------------------------------------------

processed_df = preprocess_data(df)

# ------------------------------------------------------------
# VALIDATE PREPROCESSED DATA
# ------------------------------------------------------------

validate_preprocessed_data(processed_df)

# ------------------------------------------------------------
# DISPLAY PREPROCESSING SUMMARY
# ------------------------------------------------------------

preprocessing_summary(df, processed_df)

# ------------------------------------------------------------
# DISPLAY SAMPLE OUTPUT
# ------------------------------------------------------------

print("\nSample Preprocessed Records")
print("-" * 60)

print(
    processed_df[
        ["fraudulent", "combined_text"]
    ].head(3).to_string()
)

# ------------------------------------------------------------
# SAVE PREPROCESSING REPORT
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "preprocessing_summary.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("JOBGUARD-AI PREPROCESSING REPORT\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Original records: {len(df)}\n")
    file.write(f"Processed records: {len(processed_df)}\n")

    file.write(
        f"Records removed: {len(df) - len(processed_df)}\n\n"
    )

    file.write("Final class distribution:\n")
    file.write(
        processed_df["fraudulent"]
        .value_counts()
        .to_string()
    )

    file.write("\n\nSample processed text:\n")

    file.write(
        processed_df["combined_text"].iloc[0][:1000]
    )

print("\n" + "=" * 60)
print("PREPROCESSING TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nReport saved to:")
print(report_path)

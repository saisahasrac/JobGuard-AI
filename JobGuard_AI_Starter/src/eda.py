# ============================================================
# JOBGUARD-AI
# Member 1: Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# PATH CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "fake_job_postings.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("JOBGUARD-AI: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# 1. DATASET OVERVIEW
# ------------------------------------------------------------

print("\n1. DATASET OVERVIEW")
print("-" * 60)

print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

# ------------------------------------------------------------
# 2. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n2. MISSING VALUE ANALYSIS")
print("-" * 60)

missing_data = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

missing_data = missing_data.sort_values(
    by="Missing Percentage",
    ascending=False
)

print(missing_data.round(2))

# Missing values visualization
plt.figure(figsize=(12, 7))

missing_data["Missing Percentage"].plot(kind="bar")

plt.title("Missing Values Percentage by Column")
plt.xlabel("Columns")
plt.ylabel("Missing Values (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "missing_values.jpeg", dpi=300)
plt.close()

# ------------------------------------------------------------
# 3. DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n3. DUPLICATE ANALYSIS")
print("-" * 60)

duplicate_count = df.duplicated().sum()

print("Number of Duplicate Rows:", duplicate_count)

# ------------------------------------------------------------
# 4. FRAUDULENT VS GENUINE JOB DISTRIBUTION
# ------------------------------------------------------------

print("\n4. JOB CLASS DISTRIBUTION")
print("-" * 60)

if "fraudulent" in df.columns:

    class_counts = df["fraudulent"].value_counts()

    print(class_counts)

    print("\nClass Percentages:")
    print((df["fraudulent"].value_counts(normalize=True) * 100).round(2))

    plt.figure(figsize=(8, 6))

    class_counts.plot(kind="bar")

    plt.title("Genuine vs Fraudulent Job Postings")
    plt.xlabel("Fraudulent Label")
    plt.ylabel("Number of Job Postings")
    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "class_distribution.jpeg", dpi=300)
    plt.close()

# ------------------------------------------------------------
# 5. CATEGORICAL FEATURE ANALYSIS
# ------------------------------------------------------------

print("\n5. CATEGORICAL FEATURE ANALYSIS")
print("-" * 60)

categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

for column in categorical_columns:

    if column in df.columns:

        print(f"\nDistribution of {column}:")
        print(df[column].value_counts(dropna=False).head(10))

# ------------------------------------------------------------
# 6. TEXT LENGTH ANALYSIS
# ------------------------------------------------------------

print("\n6. TEXT LENGTH ANALYSIS")
print("-" * 60)

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:

    if column in df.columns:

        text_length = df[column].fillna("").astype(str).str.len()

        print(f"\n{column}")
        print("Average Length:", round(text_length.mean(), 2))
        print("Minimum Length:", text_length.min())
        print("Maximum Length:", text_length.max())

# ------------------------------------------------------------
# 7. DESCRIPTION LENGTH DISTRIBUTION
# ------------------------------------------------------------

if "description" in df.columns:

    description_length = df["description"].fillna("").astype(str).str.len()

    plt.figure(figsize=(10, 6))

    plt.hist(description_length, bins=50)

    plt.title("Job Description Length Distribution")
    plt.xlabel("Number of Characters")
    plt.ylabel("Number of Job Postings")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "description_length.jpeg", dpi=300)
    plt.close()

# ------------------------------------------------------------
# 8. MISSING VALUES IN FRAUDULENT AND GENUINE JOBS
# ------------------------------------------------------------

print("\n8. MISSING DATA COMPARISON")
print("-" * 60)

if "fraudulent" in df.columns:

    missing_comparison = df.groupby("fraudulent").apply(
        lambda group: group.isnull().sum()
    )

    print(missing_comparison)

# ------------------------------------------------------------
# 9. TEXT LENGTH COMPARISON BY CLASS
# ------------------------------------------------------------

if "description" in df.columns and "fraudulent" in df.columns:

    analysis_df = df.copy()

    analysis_df["description_length"] = (
        analysis_df["description"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print("\nDescription Length by Job Class:")
    print(
        analysis_df.groupby("fraudulent")["description_length"]
        .describe()
    )

    plt.figure(figsize=(8, 6))

    analysis_df.boxplot(
        column="description_length",
        by="fraudulent"
    )

    plt.title("Description Length by Job Class")
    plt.suptitle("")
    plt.xlabel("Fraudulent Label")
    plt.ylabel("Description Length")

    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "description_length_by_class.jpeg", dpi=300)
    plt.close()

# ------------------------------------------------------------
# 10. SAVE EDA SUMMARY
# ------------------------------------------------------------

summary_file = OUTPUT_DIR / "eda_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("JOBGUARD-AI EDA SUMMARY\n")
    file.write("=" * 50 + "\n\n")

    file.write(f"Total Rows: {df.shape[0]}\n")
    file.write(f"Total Columns: {df.shape[1]}\n")
    file.write(f"Duplicate Rows: {duplicate_count}\n\n")

    file.write("Missing Values:\n")
    file.write(missing_data.to_string())
    file.write("\n\n")

    if "fraudulent" in df.columns:

        file.write("Class Distribution:\n")
        file.write(class_counts.to_string())
        file.write("\n\n")

        file.write("Class Percentages:\n")
        file.write(
            (df["fraudulent"].value_counts(normalize=True) * 100)
            .round(2)
            .to_string()
        )

# ------------------------------------------------------------
# COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nFiles saved in outputs folder:")
print("- missing_values.jpeg")
print("- class_distribution.jpeg")
print("- description_length.jpeg")
print("- description_length_by_class.jpeg")
print("- eda_summary.txt")

print("\nOutput Directory:", OUTPUT_DIR)

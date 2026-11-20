# ============================================================
# JOBGUARD-AI
# Data Preprocessing
# Member 1: Data Preprocessing and Data Quality
# ============================================================

import pandas as pd
import re
import html


# ============================================================
# 1. TEXT COLUMNS
# ============================================================

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "location",
    "salary_range",
]


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean text safely.

    - Handles missing values
    - Converts values to strings
    - Decodes HTML entities
    - Removes HTML tags
    - Normalizes whitespace
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove unnecessary spaces
    text = text.strip()

    return text


# ============================================================
# 3. PREPARE TEXT
# ============================================================

def prepare_text(data: pd.DataFrame) -> pd.Series:
    """
    Combine job-related text and metadata columns.

    Missing columns are treated as empty strings.
    Missing values are handled safely.

    Field prefixes are preserved.

    Returns:
        pd.Series containing combined text.
    """

    frame = data.copy()

    # Ensure all required columns exist
    for column in TEXT_COLUMNS:

        if column not in frame.columns:
            frame[column] = ""

        # Handle missing values and clean text
        frame[column] = frame[column].apply(clean_text)

    # Combine fields while preserving field names
    combined_text = frame[TEXT_COLUMNS].apply(
        lambda row: " ".join(
            f"{column}: {row[column]}"
            for column in TEXT_COLUMNS
        ),
        axis=1,
    )

    return combined_text


# ============================================================
# 4. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate records.
    """

    initial_count = len(data)

    cleaned_data = data.drop_duplicates().copy()

    removed_count = initial_count - len(cleaned_data)

    print("\nDuplicate Removal")
    print("-" * 40)
    print("Original records:", initial_count)
    print("Remaining records:", len(cleaned_data))
    print("Duplicates removed:", removed_count)

    return cleaned_data


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values safely.

    Missing target labels are removed.
    Missing text values are replaced with empty strings.
    """

    frame = data.copy()

    # Remove records with missing target labels
    if "fraudulent" in frame.columns:

        frame = frame.dropna(subset=["fraudulent"])

    # Replace missing values in text columns
    for column in TEXT_COLUMNS:

        if column in frame.columns:

            frame[column] = frame[column].fillna("")

    print("\nMissing Value Handling")
    print("-" * 40)
    print("Remaining records:", len(frame))

    return frame


# ============================================================
# 6. VALIDATE TARGET LABELS
# ============================================================

def validate_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and convert fraudulent labels.

    0 = Genuine job
    1 = Fraudulent job
    """

    frame = data.copy()

    if "fraudulent" not in frame.columns:

        raise ValueError(
            "Missing target column: fraudulent"
        )

    # Normalize label values
    frame["fraudulent"] = (
        frame["fraudulent"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Label mapping
    label_mapping = {
        "f": 0,
        "false": 0,
        "0": 0,
        "genuine": 0,
        "real": 0,

        "t": 1,
        "true": 1,
        "1": 1,
        "fraudulent": 1,
        "fake": 1,
    }

    frame["fraudulent"] = frame["fraudulent"].map(
        label_mapping
    )

    # Remove invalid labels
    frame = frame.dropna(subset=["fraudulent"])

    # Convert labels to integer
    frame["fraudulent"] = frame["fraudulent"].astype(int)

    print("\nLabel Validation")
    print("-" * 40)
    print(frame["fraudulent"].value_counts())

    return frame


# ============================================================
# 7. COMPLETE PREPROCESSING FUNCTION
# ============================================================

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data preprocessing pipeline.

    Steps:
    1. Handle missing values
    2. Remove duplicate rows
    3. Validate target labels
    4. Generate combined text

    Returns:
        Processed DataFrame.
    """

    print("=" * 60)
    print("JOBGUARD-AI PREPROCESSING")
    print("=" * 60)

    original_count = len(data)

    # Step 1: Missing values
    data = handle_missing_values(data)

    # Step 2: Duplicate removal
    data = remove_duplicates(data)

    # Step 3: Target label validation
    data = validate_labels(data)

    # Step 4: Text preparation
    data["combined_text"] = prepare_text(data)

    print("\nPreprocessing Summary")
    print("-" * 40)
    print("Original records:", original_count)
    print("Processed records:", len(data))
    print("Text column generated: combined_text")

    return data


# ============================================================
# 8. VALIDATE PREPROCESSED DATA
# ============================================================

def validate_preprocessed_data(data: pd.DataFrame):
    """
    Validate the preprocessed dataset.
    """

    print("\n" + "=" * 60)
    print("PREPROCESSED DATA VALIDATION")
    print("=" * 60)

    # Check target column
    if "fraudulent" in data.columns:

        print(
            "Missing labels:",
            data["fraudulent"].isna().sum()
        )

        print(
            "Unique labels:",
            data["fraudulent"].unique()
        )

    # Check combined text column
    if "combined_text" in data.columns:

        print(
            "Missing combined text:",
            data["combined_text"].isna().sum()
        )

        print(
            "Empty combined text:",
            (data["combined_text"].str.strip() == "").sum()
        )

        print("\nSample processed text:")
        print(data["combined_text"].iloc[0][:500])

    print("\nValidation completed successfully.")


# ============================================================
# 9. DATASET SUMMARY
# ============================================================

def preprocessing_summary(
    original_data: pd.DataFrame,
    processed_data: pd.DataFrame
):
    """
    Display a summary of preprocessing changes.
    """

    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)

    print("Original dataset size:", len(original_data))
    print("Processed dataset size:", len(processed_data))

    removed_records = (
        len(original_data) - len(processed_data)
    )

    print("Records removed:", removed_records)

    if "fraudulent" in processed_data.columns:

        print("\nFinal class distribution:")
        print(processed_data["fraudulent"].value_counts())

    print("\nPreprocessing summary completed.")

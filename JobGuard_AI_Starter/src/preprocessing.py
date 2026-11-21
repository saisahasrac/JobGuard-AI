import pandas as pd

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements", "benefits",
    "employment_type", "required_experience", "required_education",
    "industry", "function", "location", "salary_range",
]

def prepare_text(data: pd.DataFrame) -> pd.Series:
    """Combine known text/metadata columns; absent columns are treated as blank."""
    frame = data.copy()

    for column in TEXT_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    # Prefixing fields preserves a little information about where each phrase came from.
    return frame[TEXT_COLUMNS].apply(
        lambda row: " ".join(f"{col}: {row[col]}" for col in TEXT_COLUMNS),
        axis=1,
    )

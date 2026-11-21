import pandas as pd
from preprocessing import prepare_text, TEXT_COLUMNS


def test_prepare_text_returns_series():
    data = pd.DataFrame({
        "title": ["Software Engineer"],
        "description": ["Develop software"],
        "company_profile": ["ABC Company"]
    })

    result = prepare_text(data)

    assert isinstance(result, pd.Series)
    assert len(result) == 1


def test_prepare_text_contains_column_prefixes():
    data = pd.DataFrame({
        "title": ["Software Engineer"],
        "description": ["Develop software"]
    })

    result = prepare_text(data)

    assert "title:" in result.iloc[0]
    assert "description:" in result.iloc[0]


def test_prepare_text_handles_missing_columns():
    data = pd.DataFrame({
        "title": ["Data Analyst"]
    })

    result = prepare_text(data)

    assert isinstance(result, pd.Series)
    assert len(result) == 1


def test_prepare_text_handles_missing_values():
    data = pd.DataFrame({
        "title": [None],
        "description": [None]
    })

    result = prepare_text(data)

    assert result.notna().all()


def test_prepare_text_preserves_rows():
    data = pd.DataFrame({
        "title": ["Engineer", "Developer", "Analyst"]
    })

    result = prepare_text(data)

    assert len(result) == len(data)


def test_text_columns_exist():
    assert isinstance(TEXT_COLUMNS, list)
    assert len(TEXT_COLUMNS) > 0


if __name__ == "__main__":
    print("Running preprocessing checks...")

    test_prepare_text_returns_series()
    test_prepare_text_contains_column_prefixes()
    test_prepare_text_handles_missing_columns()
    test_prepare_text_handles_missing_values()
    test_prepare_text_preserves_rows()
    test_text_columns_exist()

    print("All preprocessing checks passed!")

"""Unit tests for data loading and cleaning modules."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.data_loader import (
    load_raw_data,
    clean_dataset,
    get_feature_target_split,
    EXPECTED_COLUMNS,
)


@pytest.fixture
def sample_raw_data(tmp_path) -> Path:
    """Create a temporary CSV file with edge cases including whitespace TotalCharges."""
    data = {
        "customerID": ["001-A", "002-B", "003-C", "004-D"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 0, 12, 24],
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "Yes", "No", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": [29.85, 70.35, 60.15, 20.00],
        "TotalCharges": ["29.85", " ", "720.50", "480.00"],  # Note row 1 has space
        "Churn": ["No", "Yes", "No", "No"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "sample_churn.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_load_raw_data_success(sample_raw_data):
    """Test loading valid CSV dataset."""
    df = load_raw_data(sample_raw_data)
    assert len(df) == 4
    for col in EXPECTED_COLUMNS:
        assert col in df.columns


def test_load_raw_data_missing_file():
    """Test error raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_raw_data("non_existent_file.csv")


def test_clean_dataset_whitespace_total_charges(sample_raw_data):
    """Verify that whitespace in TotalCharges is converted to 0.0."""
    raw_df = load_raw_data(sample_raw_data)
    cleaned = clean_dataset(raw_df, drop_customer_id=True)

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].dtype in [np.float64, float]
    # Row with tenure=0 and space should now be 0.0
    assert cleaned.loc[1, "TotalCharges"] == 0.0
    assert not cleaned["TotalCharges"].isna().any()


def test_clean_dataset_churn_encoding(sample_raw_data):
    """Verify that Churn is cleanly mapped to 0 and 1 integer."""
    raw_df = load_raw_data(sample_raw_data)
    cleaned = clean_dataset(raw_df)

    assert set(cleaned["Churn"].unique()).issubset({0, 1})
    assert cleaned["Churn"].tolist() == [0, 1, 0, 0]


def test_get_feature_target_split(sample_raw_data):
    """Verify feature and target separation."""
    raw_df = load_raw_data(sample_raw_data)
    cleaned = clean_dataset(raw_df)
    X, y = get_feature_target_split(cleaned, target_col="Churn")

    assert "Churn" not in X.columns
    assert len(X) == len(y) == 4
    assert isinstance(y, pd.Series)

"""Data loading and cleaning utilities for Telco Customer Churn."""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np


EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load raw dataset from CSV file.
    
    Args:
        filepath: Path to the Telco-Customer-Churn.csv file.
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty or missing key columns.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
    
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at {path} is empty.")
    
    # Check essential columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    return df


def clean_dataset(df: pd.DataFrame, drop_customer_id: bool = True) -> pd.DataFrame:
    """Clean the raw churn dataset.
    
    - Handles blank/whitespace values in TotalCharges by converting to float
      and filling with 0.0 (domain logic: customers with tenure=0 have 0 accumulated charges).
    - Converts SeniorCitizen to object/string or integer category.
    - Encodes Churn as binary integer (1 for 'Yes', 0 for 'No').
    - Drops customerID if specified.
    
    Args:
        df: Raw DataFrame.
        drop_customer_id: Whether to drop the unique customer identifier.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    
    # TotalCharges contains whitespace for new customers with tenure = 0
    if "TotalCharges" in df.columns:
        # Convert spaces / blank strings to numeric (NaN)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Customers with tenure == 0 have not been billed a full cycle yet; set TotalCharges to 0.0
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    
    # Ensure numerical columns are float/int
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
        
    # Standardize Churn column if present
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0})
        # If any unexpected values remain, raise or handle
        if df["Churn"].isna().any():
            invalid_count = df["Churn"].isna().sum()
            raise ValueError(f"Found {invalid_count} unmapped values in 'Churn' column.")
        df["Churn"] = df["Churn"].astype(int)
        
    # Drop customer ID if requested
    if drop_customer_id and "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        
    return df


def get_feature_target_split(
    df: pd.DataFrame,
    target_col: str = "Churn"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features X and target y.
    
    Args:
        df: Cleaned DataFrame.
        target_col: Name of the target variable column.
        
    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y)
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

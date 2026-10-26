"""Unit tests for feature preprocessing pipeline."""

import pandas as pd
import numpy as np
from src.preprocessing import (
    build_preprocessor,
    get_feature_names,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def test_build_preprocessor():
    """Verify preprocessor construction and output features."""
    preprocessor = build_preprocessor()
    assert preprocessor is not None

    # Create dummy data matching schema
    data = {
        "tenure": [1, 24, 60],
        "MonthlyCharges": [30.0, 75.5, 90.0],
        "TotalCharges": [30.0, 1800.0, 5400.0],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
    }
    df = pd.DataFrame(data)

    transformed = preprocessor.fit_transform(df)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 3
    # Check transformed output is not empty
    assert transformed.shape[1] > len(NUMERICAL_FEATURES)

    feature_names = get_feature_names(preprocessor)
    assert len(feature_names) == transformed.shape[1]

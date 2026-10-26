"""Data preprocessing pipeline for Telco Customer Churn."""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for preprocessing numerical and categorical features.
    
    Returns:
        ColumnTransformer: Preprocessing transformer.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Retrieve output feature names from a fitted ColumnTransformer.
    
    Args:
        preprocessor: Fitted ColumnTransformer.
        
    Returns:
        List[str]: Names of transformed features.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback if get_feature_names_out fails
        features = list(NUMERICAL_FEATURES)
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        if hasattr(cat_encoder, "get_feature_names_out"):
            cat_features = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
            features.extend(cat_features)
        return features

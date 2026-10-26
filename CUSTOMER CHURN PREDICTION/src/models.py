"""Model training, serialization, and feature importance utilities."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

from src.preprocessing import build_preprocessor, get_feature_names


def build_model_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Build unfitted pipelines for Logistic Regression and Random Forest.
    
    Args:
        random_state: Random seed for reproducibility.
        
    Returns:
        Dict[str, Pipeline]: Map of model name to pipeline.
    """
    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=9,
                    min_samples_split=6,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic_regression,
        "Random Forest": random_forest,
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Dict[str, Pipeline]:
    """Train both Logistic Regression and Random Forest models on training data.
    
    Args:
        X_train: Training features.
        y_train: Training target.
        random_state: Random seed.
        
    Returns:
        Dict[str, Pipeline]: Fitted pipelines.
    """
    pipelines = build_model_pipelines(random_state=random_state)
    fitted = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted


def extract_feature_importances(
    pipeline: Pipeline,
    model_name: str
) -> pd.DataFrame:
    """Extract and sort feature importances or coefficients.
    
    Args:
        pipeline: Fitted pipeline with 'preprocessor' and 'classifier'.
        model_name: Name of the model.
        
    Returns:
        pd.DataFrame: Table with feature names and importance values.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = get_feature_names(preprocessor)

    if isinstance(classifier, LogisticRegression):
        coefs = classifier.coef_[0]
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefs,
            "abs_importance": np.abs(coefs),
            "effect": np.where(coefs > 0, "Increases Churn", "Decreases Churn")
        }).sort_values(by="abs_importance", ascending=False).reset_index(drop=True)
        return df_importance
    elif isinstance(classifier, RandomForestClassifier):
        importances = classifier.feature_importances_
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
            "abs_importance": importances,
            "effect": "Relative Weight"
        }).sort_values(by="abs_importance", ascending=False).reset_index(drop=True)
        return df_importance
    else:
        raise ValueError(f"Unsupported classifier type: {type(classifier)}")


def save_pipeline(pipeline: Pipeline, filepath: str | Path) -> Path:
    """Save pipeline to disk via Joblib.
    
    Args:
        pipeline: Fitted pipeline.
        filepath: Destination path.
        
    Returns:
        Path: Path to saved file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(filepath: str | Path) -> Pipeline:
    """Load trained pipeline from disk via Joblib.
    
    Args:
        filepath: Path to saved model file.
        
    Returns:
        Pipeline: Loaded pipeline.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Trained model not found at: {path.resolve()}")
    return joblib.load(path)

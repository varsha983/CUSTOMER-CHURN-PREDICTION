"""Unit tests for trained model pipeline inference and serialization."""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models import build_model_pipelines, load_pipeline, extract_feature_importances
from src.preprocessing import ALL_FEATURE_COLUMNS


@pytest.fixture
def synthetic_train_test_data():
    """Generate consistent synthetic dataframe for quick pipeline testing."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "tenure": np.random.randint(0, 72, size=n),
        "MonthlyCharges": np.random.uniform(20.0, 110.0, size=n),
        "TotalCharges": np.random.uniform(20.0, 7000.0, size=n),
        "gender": np.random.choice(["Male", "Female"], size=n),
        "SeniorCitizen": np.random.choice([0, 1], size=n),
        "Partner": np.random.choice(["Yes", "No"], size=n),
        "Dependents": np.random.choice(["Yes", "No"], size=n),
        "PhoneService": np.random.choice(["Yes", "No"], size=n),
        "MultipleLines": np.random.choice(["No", "Yes", "No phone service"], size=n),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n),
        "OnlineSecurity": np.random.choice(["No", "Yes", "No internet service"], size=n),
        "OnlineBackup": np.random.choice(["No", "Yes", "No internet service"], size=n),
        "DeviceProtection": np.random.choice(["No", "Yes", "No internet service"], size=n),
        "TechSupport": np.random.choice(["No", "Yes", "No internet service"], size=n),
        "StreamingTV": np.random.choice(["No", "Yes", "No internet service"], size=n),
        "StreamingMovies": np.random.choice(["No", "Yes", "No internet service"], size=n),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=n),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ], size=n),
    })
    y = np.random.choice([0, 1], size=n, p=[0.73, 0.27])
    return df, pd.Series(y)


def test_model_pipeline_fit_predict(synthetic_train_test_data):
    """Test training and predicting with both Logistic Regression and Random Forest."""
    X, y = synthetic_train_test_data
    pipelines = build_model_pipelines(random_state=42)

    for name, pipeline in pipelines.items():
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        probs = pipeline.predict_proba(X)

        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})
        assert probs.shape == (len(X), 2)
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_pipeline_handles_unseen_category(synthetic_train_test_data):
    """Ensure pipeline handles unseen categorical values gracefully without throwing errors."""
    X, y = synthetic_train_test_data
    pipelines = build_model_pipelines(random_state=42)
    rf = pipelines["Random Forest"]
    rf.fit(X, y)

    # Inject an unseen category value
    unseen_sample = X.iloc[[0]].copy()
    unseen_sample.loc[0, "PaymentMethod"] = "Crypto / Unknown Currency"
    unseen_sample.loc[0, "Contract"] = "Decade Plan"

    prob = rf.predict_proba(unseen_sample)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0


def test_production_model_loading_and_inference():
    """Verify that the saved best_model.joblib loads and generates valid predictions."""
    model_path = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"
    assert model_path.is_file(), "best_model.joblib must exist."

    model = load_pipeline(model_path)
    assert isinstance(model, Pipeline)

    # Test sample profile
    test_customer = pd.DataFrame([{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 171.00,
    }])

    prob = model.predict_proba(test_customer)[0, 1]
    pred = model.predict(test_customer)[0]

    assert 0.0 <= prob <= 1.0
    assert pred in [0, 1]


def test_extract_feature_importances(synthetic_train_test_data):
    """Verify feature importance extraction returns sorted valid dataframe."""
    X, y = synthetic_train_test_data
    pipelines = build_model_pipelines(random_state=42)

    for name, pipeline in pipelines.items():
        pipeline.fit(X, y)
        imp_df = extract_feature_importances(pipeline, name)
        assert "feature" in imp_df.columns
        assert "abs_importance" in imp_df.columns
        assert len(imp_df) > 0
        # Ensure descending sort
        assert (imp_df["abs_importance"].diff().dropna() <= 1e-6).all()

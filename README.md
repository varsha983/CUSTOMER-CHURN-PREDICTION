# Telco Customer Churn Prediction & Retention Analytics

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) system designed to identify telecommunication customers at risk of churn and deliver actionable retention insights through an interactive dashboard.

---

## 1. Project Overview & Problem Statement

### Objective
In subscription and telecommunication businesses, customer retention is significantly more cost-effective than customer acquisition. The primary goal of this project is to build an automated, reliable, and interpretable classification pipeline that:
1. Predicts the likelihood of customer churn (`Churn = Yes` or `No`).
2. Isolates the strongest behavioral, financial, and contractual drivers influencing customer departure.
3. Provides an interactive decision-support dashboard for retention teams to simulate customer profiles and evaluate risk tiers.

### Problem Formulation
Customer attrition exhibits class imbalance (~26.5% churn vs. ~73.5% retention). A naive model predicting "No Churn" achieves 73.5% accuracy while missing 100% of departing customers. Therefore, this project focuses on **Recall** and **ROC-AUC** metrics, balancing false positives against the business risk of failing to identify at-risk accounts.

---

## 2. Dataset Information & Source

- **Dataset**: IBM Telco Customer Churn
- **Source Repository**: [IBM / telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d)
- **Direct Download**: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
- **License**: Apache License 2.0
- **Size**: 7,043 customer profiles, 21 attributes (20 predictor features + 1 target variable).

### Data Quality & Preprocessing Highlights
- **Whitespace Handling**: The raw dataset contains 11 records with blank space `' '` values in `TotalCharges`. All 11 records correspond to customers with `tenure = 0` (new accounts that have not completed a billing cycle). These values are cleanly imputed with `0.0`, preserving all customer rows without dropping data.
- **Pipeline Integrity & Leakage Prevention**: Numerical standardization (`StandardScaler`) and categorical encoding (`OneHotEncoder`) are bundled into a Scikit-Learn `ColumnTransformer` fitted strictly on training data (`X_train`), eliminating data leakage.
- **Unseen Category Robustness**: The pipeline handles unknown category values at inference time via `handle_unknown='ignore'`.

---

## 3. Technology Stack

- **Language**: Python 3.12 (macOS Apple Silicon & CPU compatible)
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (`Pipeline`, `ColumnTransformer`, `LogisticRegression`, `RandomForestClassifier`)
- **Serialization**: `joblib`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Interactive Web App**: `streamlit`
- **Testing**: `pytest`

---

## 4. Repository Structure

```
CUSTOMER CHURN PREDICTION/
│
├── .gitignore                   # Excludes .venv, caches, logs, .DS_Store
├── .streamlit/
│   └── config.toml              # Streamlit server and telemetry configurations
├── requirements.txt             # Locked project dependencies
├── README.md                    # Detailed documentation and reproduction steps
├── train.py                     # End-to-end training, EDA, and evaluation script
├── app.py                       # Streamlit web application & prediction tool
│
├── data/
│   ├── README.md                # Dataset attribution and schema specifications
│   └── Telco-Customer-Churn.csv # Cleaned raw dataset
│
├── models/
│   ├── best_model.joblib        # Production-selected model pipeline (Random Forest)
│   ├── random_forest_pipeline.joblib
│   └── logistic_regression_pipeline.joblib
│
├── reports/
│   ├── eda_summary.md           # Detailed EDA findings report
│   ├── evaluation_summary.json  # Exact quantitative evaluation results
│   ├── feature_importance_logistic_regression.csv
│   ├── feature_importance_random_forest.csv
│   └── figures/                 # Exported publication-ready visualizations
│       ├── confusion_matrix.png
│       ├── eda_charges_distribution.png
│       ├── eda_churn_by_contract.png
│       ├── eda_churn_distribution.png
│       ├── eda_correlation_matrix.png
│       ├── eda_internet_service.png
│       ├── eda_payment_method.png
│       ├── eda_tenure_distribution.png
│       ├── feature_importance_logistic_regression.png
│       ├── feature_importance_random_forest.png
│       └── roc_curve.png
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Ingestion, validation, and data cleaning
│   ├── eda.py                   # Chart generation and statistical reporting
│   ├── preprocessing.py         # Leakage-free Scikit-Learn ColumnTransformer
│   ├── models.py                # Pipeline construction, training, and feature extraction
│   └── evaluate.py              # Performance metrics, ROC, and confusion matrices
│
└── tests/
    ├── __init__.py
    ├── test_data_loader.py      # Unit tests for loading and cleaning
    ├── test_preprocessing.py    # Unit tests for preprocessing transformer
    └── test_pipeline.py         # Unit tests for inference, edge cases, and serialization
```

---

## 5. Machine Learning Evaluation & Results

Both models were evaluated on an identical **stratified holdout test set** (20% of dataset = 1,409 customers; Churn rate = 26.54%). Class weighting was set to `balanced` to address class skew.

### Model Comparison Table (Test Set)

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** *(Selected)* | **75.37%** | **52.37%** | **79.68%** | **0.6320** | **0.8422** |
| **Logistic Regression** | 73.81% | 50.43% | 78.34% | 0.6136 | 0.8415 |

### Confusion Matrix Breakdown (Test Set: 1,409 customers)
- **Random Forest**:
  - True Negatives (Retained correctly identified): **764**
  - False Positives (Retained flagged as at-risk): **271**
  - False Negatives (Churners missed): **76**
  - True Positives (Churners caught): **298**
- **Recall on Churners**: **79.68%** (captures nearly 8 out of every 10 departing customers).

### Production Selection Rationale
**Random Forest Classifier** was selected as the production model (`models/best_model.joblib`) because it achieved the highest ROC-AUC (**0.8422**), the highest Recall (**79.68%**), and superior F1-Score (**0.6320**), while naturally capturing non-linear feature interactions (such as the compounding risk of short tenure with high monthly charges).

---

## 6. Key Business Findings & Predictive Drivers

1. **Contract Type Dominance**:
   - Month-to-month contracts exhibit an alarming **42.7%** churn rate.
   - One-year contracts drop to **11.3%** churn.
   - Two-year contracts drop to **2.8%** churn.
2. **The 6-Month Onboarding Window**:
   - Customers in months 1–6 face the highest attrition hazard.
   - The average tenure of churned customers is **18.0 months**, compared to **37.7 months** for retained accounts.
3. **Payment Method Friction**:
   - Customers paying with **Electronic Check** churn at **45.3%**, whereas customers enrolled in **Credit Card (Automatic)** churn at only **15.2%**.
4. **Technology Type Impact**:
   - **Fiber optic** subscribers experience higher churn (**41.9%**) than **DSL** subscribers (**18.9%**), driven by higher pricing combined with lower tech-support adoption.

---

## 7. How to Reproduce & Run

### A. Run Automated Unit Tests
```bash
# Run all 10 unit tests
.venv/bin/pytest -v
```

### B. Train Models & Generate Reports
```bash
# Executes data loading, cleaning, EDA figure generation, model training, and evaluation
.venv/bin/python train.py
```
*Outputs will be saved in `models/`, `reports/figures/`, and `reports/`.*

### C. Launch Interactive Streamlit Dashboard
```bash
# Launch the web application
.venv/bin/streamlit run app.py
```
Open your browser at **http://localhost:8501** to access the dashboard.

---

## 8. Dashboard Features

1. **Executive Overview**: Real-time KPI metric tiles, high-level business findings, and sample dataset view.
2. **Customer Segment Explorer**: Multidimensional filtering across contracts, internet services, payment types, and senior citizen status with instantaneous cohort size and churn rate delta calculations.
3. **Model Performance & Benchmarks**: Genuine test set metric tables, ROC curves, confusion matrices, and Gini importance / logistic regression log-odds rankings.
4. **Real-Time Customer Churn Predictor**:
   - Preset risk profiles: *New Month-to-Month (High Risk)*, *2-Year Long-Term (Low Risk)*, and *Mid-Tenure Standard (Average Risk)*.
   - Dynamic probability gauge and risk tier classification (*Critical*, *Moderate*, *Low*).
   - Automated root-cause driver identification.
   - Actionable retention recommendations.

---

## 9. Limitations & Ethical Considerations

- **Probabilistic Risk**: Model predictions are statistical estimates; they indicate probability, not certainty.
- **Precision vs. Recall Tradeoff**: Prioritizing recall (~80%) to avoid missing churners produces false positives (~48% of flagged customers would not have churned). Retention strategies should focus on value-add incentives (e.g., support vouchers, plan optimizations) rather than aggressive interventions.
- **Historical Nature**: The dataset represents a historical snapshot and does not capture macroeconomic shifts or real-time competitor pricing changes without ongoing retraining.

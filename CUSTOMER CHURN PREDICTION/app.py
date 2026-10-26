"""Streamlit Dashboard for Customer Churn Prediction."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import joblib

from src.data_loader import load_raw_data, clean_dataset
from src.preprocessing import ALL_FEATURE_COLUMNS


# Page configuration
st.set_page_config(
    page_title="Telco Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METRICS_PATH = BASE_DIR / "reports" / "evaluation_summary.json"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
RF_IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance_random_forest.csv"
LR_IMPORTANCE_PATH = BASE_DIR / "reports" / "feature_importance_logistic_regression.csv"


@st.cache_data
def get_dataset():
    raw_df = load_raw_data(DATA_PATH)
    clean_df = clean_dataset(raw_df, drop_customer_id=False)
    return raw_df, clean_df


@st.cache_resource
def get_model():
    if not MODEL_PATH.is_file():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_evaluation_metrics():
    if not METRICS_PATH.is_file():
        return None
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def main():
    st.title("📊 Telco Customer Churn Intelligence & Prediction Hub")
    st.caption("Internship Portfolio Project | Machine Learning & Exploratory Data Analysis")

    try:
        raw_df, clean_df = get_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    model = get_model()
    eval_metrics = get_evaluation_metrics()

    # Sidebar Navigation & Model Info
    st.sidebar.header("🧭 Navigation")
    section = st.sidebar.radio(
        "Jump to section:",
        [
            "Executive Overview",
            "Customer Segment Explorer",
            "Model Performance & Benchmarks",
            "Interactive Churn Predictor",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📌 Project Details")
    st.sidebar.markdown(
        """
        - **Dataset**: IBM Telco Customer Churn
        - **Total Records**: 7,043
        - **Target**: Churn (Yes / No)
        - **Production Model**: Random Forest (Stratified & Balanced)
        - **Test ROC-AUC**: 0.8422
        """
    )
    st.sidebar.info("💡 **Tip**: Use the **Interactive Churn Predictor** tab to test simulated customer risk scenarios.")

    # -------------------------------------------------------------
    # 1. Executive Overview
    # -------------------------------------------------------------
    if section == "Executive Overview":
        st.header("Executive Summary & Dataset Vital Statistics")
        st.markdown(
            """
            This end-to-end data science project analyzes telecommunication customer attrition patterns,
            evaluates multiple machine learning models trained to handle class imbalance, and provides real-time
            risk scoring to guide proactive retention campaigns.
            """
        )

        total_customers = len(clean_df)
        churn_count = int(clean_df["Churn"].sum())
        retained_count = total_customers - churn_count
        churn_rate = (churn_count / total_customers) * 100
        avg_monthly = clean_df["MonthlyCharges"].mean()
        avg_tenure = clean_df["tenure"].mean()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Churned Cohort", f"{churn_count:,}", delta=f"-{churn_rate:.1f}%", delta_color="inverse")
        col4.metric("Avg. Monthly Spend", f"${avg_monthly:.2f}")
        col5.metric("Avg. Tenure", f"{avg_tenure:.1f} mos")

        st.markdown("---")
        st.subheader("Key Business Insights At A Glance")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.warning("⚠️ **Contract Vulnerability**")
            st.markdown(
                """
                **Month-to-month contracts** carry a **42.7%** churn rate, compared to **11.3%** for 1-year and **2.8%** for 2-year contracts.
                """
            )
        with c2:
            st.error("🚨 **The 6-Month Hazard Window**")
            st.markdown(
                """
                Customer attrition is concentrated in early tenure. Churned customers have an average tenure of **18.0 months** vs **37.7 months** for retained accounts.
                """
            )
        with c3:
            st.info("💳 **Payment Method Friction**")
            st.markdown(
                """
                Customers paying via **Electronic Check** experience a **45.3%** churn rate, more than double automated credit card payments (**15.2%**).
                """
            )

        st.markdown("---")
        st.subheader("Dataset Sample & Schema Explorer")
        st.dataframe(
            clean_df.head(10),
            use_container_width=True,
            column_config={
                "MonthlyCharges": st.column_config.NumberColumn(format="$%.2f"),
                "TotalCharges": st.column_config.NumberColumn(format="$%.2f"),
                "Churn": st.column_config.CheckboxColumn(help="1 = Churned, 0 = Retained"),
            },
        )

        with st.expander("ℹ️ Data Preprocessing & Quality Notice"):
            st.markdown(
                """
                - **TotalCharges Imputation**: The raw dataset contains 11 records with whitespace (`' '`) in `TotalCharges`. All 11 records correspond to customers with `tenure = 0` (new accounts). These were imputed with `0.0` to preserve mathematical consistency without dropping valuable customer records.
                - **Class Imbalance**: Churn occurs in 26.54% of records. Class-balanced weighting was utilized during machine learning training to ensure high recall for at-risk accounts.
                """
            )

    # -------------------------------------------------------------
    # 2. Customer Segment Explorer
    # -------------------------------------------------------------
    elif section == "Customer Segment Explorer":
        st.header("🔍 Customer Segment & Exploratory Data Analysis")
        st.markdown("Interact with demographic and service slices to investigate cohort vulnerability.")

        # Interactive segment filter
        st.subheader("Interactive Cohort Filter")
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            selected_contract = st.multiselect(
                "Contract Type:",
                options=list(clean_df["Contract"].unique()),
                default=list(clean_df["Contract"].unique()),
            )
        with f_col2:
            selected_internet = st.multiselect(
                "Internet Service:",
                options=list(clean_df["InternetService"].unique()),
                default=list(clean_df["InternetService"].unique()),
            )
        with f_col3:
            selected_payment = st.multiselect(
                "Payment Method:",
                options=list(clean_df["PaymentMethod"].unique()),
                default=list(clean_df["PaymentMethod"].unique()),
            )
        with f_col4:
            selected_senior = st.selectbox(
                "Senior Citizen:",
                options=["All", "Non-Senior (0)", "Senior Citizen (1)"],
            )

        # Apply filtering
        filtered_df = clean_df[
            (clean_df["Contract"].isin(selected_contract))
            & (clean_df["InternetService"].isin(selected_internet))
            & (clean_df["PaymentMethod"].isin(selected_payment))
        ]
        if selected_senior == "Non-Senior (0)":
            filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 0]
        elif selected_senior == "Senior Citizen (1)":
            filtered_df = filtered_df[filtered_df["SeniorCitizen"] == 1]

        seg_total = len(filtered_df)
        if seg_total > 0:
            seg_churn_count = int(filtered_df["Churn"].sum())
            seg_churn_rate = (seg_churn_count / seg_total) * 100
            seg_avg_charges = filtered_df["MonthlyCharges"].mean()
            seg_avg_tenure = filtered_df["tenure"].mean()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Segment Size", f"{seg_total:,} ({seg_total/len(clean_df)*100:.1f}% of base)")
            baseline_churn = clean_df["Churn"].mean() * 100
            diff = seg_churn_rate - baseline_churn
            m2.metric(
                "Segment Churn Rate",
                f"{seg_churn_rate:.1f}%",
                delta=f"{diff:+.1f}% vs baseline",
                delta_color="inverse",
            )
            m3.metric("Segment Avg Monthly", f"${seg_avg_charges:.2f}")
            m4.metric("Segment Avg Tenure", f"{seg_avg_tenure:.1f} mos")
        else:
            st.warning("No records match the selected filter combination.")

        st.markdown("---")
        st.subheader("Publication-Ready EDA Visualizations")

        eda_tabs = st.tabs([
            "Overall & Contract",
            "Tenure Dynamics",
            "Monthly Spend",
            "Internet & Services",
            "Payment Methods",
            "Feature Correlation",
        ])

        with eda_tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.image(str(FIGURES_DIR / "eda_churn_distribution.png"), caption="Target Class Distribution")
            with c2:
                st.image(str(FIGURES_DIR / "eda_churn_by_contract.png"), caption="Churn Rate by Contract Duration")

        with eda_tabs[1]:
            st.image(str(FIGURES_DIR / "eda_tenure_distribution.png"), use_container_width=True)
            st.caption("Kernel Density Estimate (KDE): Peak hazard rate appears during the first 1-6 months.")

        with eda_tabs[2]:
            st.image(str(FIGURES_DIR / "eda_charges_distribution.png"), use_container_width=True)
            st.caption("Distribution of Monthly Charges: Churned customers show a higher median monthly commitment.")

        with eda_tabs[3]:
            st.image(str(FIGURES_DIR / "eda_internet_service.png"), use_container_width=True)
            st.caption("Churn Across Internet Providers: Fiber optic customers experience highest attrition (41.9%).")

        with eda_tabs[4]:
            st.image(str(FIGURES_DIR / "eda_payment_method.png"), use_container_width=True)
            st.caption("Payment Methods: Automated bank/card payment methods have significantly lower churn.")

        with eda_tabs[5]:
            st.image(str(FIGURES_DIR / "eda_correlation_matrix.png"), use_container_width=True)
            st.caption("Correlation Heatmap: Demonstrates strong correlation between tenure and TotalCharges.")

    # -------------------------------------------------------------
    # 3. Model Performance & Benchmarks
    # -------------------------------------------------------------
    elif section == "Model Performance & Benchmarks":
        st.header("🤖 Machine Learning Model Benchmarks & Interpretability")
        st.markdown(
            """
            Two algorithms were rigorously trained using a stratified 80/20 train/test split.
            Data leakage was prevented by executing feature scaling and one-hot encoding strictly within Scikit-Learn pipelines.
            """
        )

        if eval_metrics:
            comp_rows = []
            for name in ["Random Forest", "Logistic Regression"]:
                if name in eval_metrics:
                    m = eval_metrics[name]
                    comp_rows.append({
                        "Model": name,
                        "Accuracy": f"{m['accuracy']:.4f}",
                        "Precision": f"{m['precision']:.4f}",
                        "Recall (Sensitivity)": f"{m['recall']:.4f}",
                        "F1-Score": f"{m['f1_score']:.4f}",
                        "ROC-AUC": f"{m['roc_auc']:.4f}",
                    })
            st.subheader("Holdout Test Set Performance Comparison")
            st.table(pd.DataFrame(comp_rows))

            st.success(
                f"🏆 **Selected Production Model: {eval_metrics.get('selected_best_model', 'Random Forest')}**\n\n"
                f"{eval_metrics.get('selection_rationale', '')}"
            )

        st.markdown("---")
        st.subheader("Diagnostic Curves & Confusion Matrices")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if (FIGURES_DIR / "roc_curve.png").is_file():
                st.image(str(FIGURES_DIR / "roc_curve.png"), caption="ROC Curves Comparison")
        with p_col2:
            if (FIGURES_DIR / "confusion_matrix.png").is_file():
                st.image(str(FIGURES_DIR / "confusion_matrix.png"), caption="Confusion Matrices (Holdout Set)")

        st.markdown("---")
        st.subheader("Predictive Feature Drivers")
        imp_tab1, imp_tab2 = st.tabs(["Random Forest (Gini Importance)", "Logistic Regression (Log-Odds)"])

        with imp_tab1:
            if (FIGURES_DIR / "feature_importance_random_forest.png").is_file():
                st.image(str(FIGURES_DIR / "feature_importance_random_forest.png"), use_container_width=True)
            if RF_IMPORTANCE_PATH.is_file():
                rf_df = pd.read_csv(RF_IMPORTANCE_PATH).head(10)
                st.dataframe(rf_df, use_container_width=True)

        with imp_tab2:
            if (FIGURES_DIR / "feature_importance_logistic_regression.png").is_file():
                st.image(str(FIGURES_DIR / "feature_importance_logistic_regression.png"), use_container_width=True)
            if LR_IMPORTANCE_PATH.is_file():
                lr_df = pd.read_csv(LR_IMPORTANCE_PATH).head(10)
                st.dataframe(lr_df, use_container_width=True)

    # -------------------------------------------------------------
    # 4. Interactive Churn Predictor
    # -------------------------------------------------------------
    elif section == "Interactive Churn Predictor":
        st.header("🔮 Real-Time Customer Churn Risk Predictor")
        st.markdown(
            "Configure a customer's contract, tenure, and services to evaluate their estimated churn probability."
        )

        if model is None:
            st.error("Trained model pipeline not found. Please run `python train.py` first.")
            st.stop()

        # Quick preset buttons
        st.markdown("##### ⚡ Quick Load Preset Profiles")
        preset_cols = st.columns(4)
        preset_clicked = None
        with preset_cols[0]:
            if st.button("🚨 New Month-to-Month (High Risk)"):
                preset_clicked = "high_risk"
        with preset_cols[1]:
            if st.button("🛡️ 2-Year Long-Term (Low Risk)"):
                preset_clicked = "low_risk"
        with preset_cols[2]:
            if st.button("⚖️ Mid-Tenure Standard (Average Risk)"):
                preset_clicked = "medium_risk"
        with preset_cols[3]:
            if st.button("🔄 Reset Defaults"):
                preset_clicked = "default"

        # Determine default values based on presets
        defaults = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
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
            "MonthlyCharges": 89.50,
            "TotalCharges": 268.50,
        }

        if preset_clicked == "low_risk":
            defaults.update({
                "tenure": 48,
                "Contract": "Two year",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "TechSupport": "Yes",
                "PaymentMethod": "Credit card (automatic)",
                "MonthlyCharges": 55.00,
                "TotalCharges": 2640.00,
            })
        elif preset_clicked == "medium_risk":
            defaults.update({
                "tenure": 24,
                "Contract": "One year",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "TechSupport": "No",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 75.00,
                "TotalCharges": 1800.00,
            })

        # Form
        with st.form("churn_prediction_form"):
            st.subheader("1. Account & Contract Information")
            c1, c2, c3 = st.columns(3)
            with c1:
                contract = st.selectbox(
                    "Contract Duration",
                    options=["Month-to-month", "One year", "Two year"],
                    index=["Month-to-month", "One year", "Two year"].index(defaults["Contract"]),
                )
                tenure = st.number_input(
                    "Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=int(defaults["tenure"]),
                    help="Number of months customer has been with the company.",
                )
            with c2:
                payment_method = st.selectbox(
                    "Payment Method",
                    options=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ],
                    index=[
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ].index(defaults["PaymentMethod"]),
                )
                paperless = st.selectbox(
                    "Paperless Billing",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(defaults["PaperlessBilling"]),
                )
            with c3:
                monthly_charges = st.number_input(
                    "Monthly Charges ($ USD)",
                    min_value=18.0,
                    max_value=125.0,
                    value=float(defaults["MonthlyCharges"]),
                    step=0.5,
                )
                auto_total = float(tenure * monthly_charges) if tenure > 0 else 0.0
                total_charges = st.number_input(
                    "Total Cumulative Charges ($ USD)",
                    min_value=0.0,
                    max_value=9000.0,
                    value=float(defaults["TotalCharges"]),
                    step=5.0,
                    help=f"Expected estimate: ~${auto_total:.2f} based on tenure.",
                )

            st.markdown("---")
            st.subheader("2. Demographics")
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                gender = st.selectbox(
                    "Gender",
                    options=["Male", "Female"],
                    index=["Male", "Female"].index(defaults["gender"]),
                )
            with d2:
                senior = st.selectbox(
                    "Senior Citizen",
                    options=[0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                    index=[0, 1].index(defaults["SeniorCitizen"]),
                )
            with d3:
                partner = st.selectbox(
                    "Partner",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(defaults["Partner"]),
                )
            with d4:
                dependents = st.selectbox(
                    "Dependents",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(defaults["Dependents"]),
                )

            st.markdown("---")
            st.subheader("3. Subscribed Services")
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                phone_service = st.selectbox(
                    "Phone Service",
                    options=["Yes", "No"],
                    index=["Yes", "No"].index(defaults["PhoneService"]),
                )
                mult_lines = st.selectbox(
                    "Multiple Lines",
                    options=["No", "Yes", "No phone service"],
                    index=["No", "Yes", "No phone service"].index(defaults["MultipleLines"]),
                )
            with s2:
                internet_service = st.selectbox(
                    "Internet Service",
                    options=["DSL", "Fiber optic", "No"],
                    index=["DSL", "Fiber optic", "No"].index(defaults["InternetService"]),
                )
                online_sec = st.selectbox(
                    "Online Security",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(defaults["OnlineSecurity"]),
                )
            with s3:
                online_backup = st.selectbox(
                    "Online Backup",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(defaults["OnlineBackup"]),
                )
                dev_protect = st.selectbox(
                    "Device Protection",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(defaults["DeviceProtection"]),
                )
            with s4:
                tech_support = st.selectbox(
                    "Tech Support",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(defaults["TechSupport"]),
                )
                stream_tv = st.selectbox(
                    "Streaming TV",
                    options=["No", "Yes", "No internet service"],
                    index=["No", "Yes", "No internet service"].index(defaults["StreamingTV"]),
                )

            stream_mov = st.selectbox(
                "Streaming Movies",
                options=["No", "Yes", "No internet service"],
                index=["No", "Yes", "No internet service"].index(defaults["StreamingMovies"]),
            )

            submit_prediction = st.form_submit_button("⚡ Run Churn Risk Analysis", use_container_width=True)

        if submit_prediction:
            input_dict = {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": mult_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_sec,
                "OnlineBackup": online_backup,
                "DeviceProtection": dev_protect,
                "TechSupport": tech_support,
                "StreamingTV": stream_tv,
                "StreamingMovies": stream_mov,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }

            input_df = pd.DataFrame([input_dict])

            try:
                # Predict probability
                churn_prob = float(model.predict_proba(input_df)[0, 1])
                pred_label = int(model.predict(input_df)[0])
            except Exception as e:
                st.error(f"Prediction inference failed: {e}")
                st.stop()

            st.markdown("---")
            st.subheader("🎯 Prediction Output & Risk Assessment")

            out_col1, out_col2 = st.columns([1, 2])

            with out_col1:
                prob_pct = churn_prob * 100
                st.metric("Estimated Churn Probability", f"{prob_pct:.1f}%")
                if churn_prob >= 0.50:
                    st.error("🚨 **Classification: LIKELY TO CHURN (At Risk)**")
                else:
                    st.success("✅ **Classification: LIKELY TO RETAIN (Stable)**")

                st.progress(churn_prob)

            with out_col2:
                if churn_prob >= 0.70:
                    risk_tier = "Critical Risk"
                    st.error(f"**Customer Risk Tier**: {risk_tier}")
                    st.markdown(
                        """
                        **Action Recommended**: Immediate proactive retention intervention. Offer 12-month loyalty bundle, tech support voucher, or automated billing discount.
                        """
                    )
                elif churn_prob >= 0.45:
                    risk_tier = "Moderate Risk"
                    st.warning(f"**Customer Risk Tier**: {risk_tier}")
                    st.markdown(
                        """
                        **Action Recommended**: Customer shows elevated sensitivity. Provide targeted engagement check-in and explore long-term discount upgrade.
                        """
                    )
                else:
                    risk_tier = "Low Risk / Healthy Account"
                    st.success(f"**Customer Risk Tier**: {risk_tier}")
                    st.markdown(
                        """
                        **Action Recommended**: Maintain standard service delivery. Customer exhibits strong stability indicators (tenure or contract protection).
                        """
                    )

            st.markdown("#### 🔍 Primary Contributing Factors for this Customer")
            drivers = []
            if contract == "Month-to-month":
                drivers.append("⚠️ **Month-to-Month Contract**: Strongest positive driver of customer churn in the model.")
            elif contract in ["One year", "Two year"]:
                drivers.append(f"🛡️ **{contract} Contract**: Strong protective barrier against churn.")

            if tenure < 6:
                drivers.append("⚠️ **Early Tenure (<6 months)**: Customer is inside the high-hazard onboarding window.")
            elif tenure >= 24:
                drivers.append(f"🛡️ **Long Tenure ({tenure} months)**: Established loyalty reduces churn risk.")

            if payment_method == "Electronic check":
                drivers.append("⚠️ **Electronic Check Payment**: Statistically correlated with higher payment friction and churn.")
            elif "automatic" in payment_method:
                drivers.append(f"🛡️ **Automatic Payment ({payment_method})**: Lowers churn risk via frictionless renewals.")

            if internet_service == "Fiber optic" and tech_support == "No":
                drivers.append("⚠️ **Fiber Optic without Tech Support**: High monthly bill without tech support increases churn odds.")

            for d in drivers:
                st.markdown(f"- {d}")

            st.markdown("---")
            with st.expander("⚠️ Model Limitations & Responsible AI Disclaimer"):
                st.markdown(
                    """
                    1. **Probabilistic Scoring**: Machine learning outputs are statistical risk estimates, not deterministic certainty.
                    2. **False Positives & Negatives**: As indicated by the test set recall (~80%) and precision (~52%), some customers predicted to churn would stay organically, and vice-versa.
                    3. **Decision Support**: This tool is designed to augment human retention specialists, not replace customer conversations.
                    4. **Static Snapshot**: Model predictions are based on historical IBM Telco data and do not reflect macroeconomic shocks or real-time competitor price cuts without periodic retraining.
                    """
                )


if __name__ == "__main__":
    main()

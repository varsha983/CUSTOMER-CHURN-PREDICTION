"""End-to-end training and evaluation script for Customer Churn Prediction."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_raw_data, clean_dataset, get_feature_target_split
from src.eda import perform_eda
from src.models import train_models, extract_feature_importances, save_pipeline
from src.evaluate import (
    evaluate_model,
    generate_comparison_table,
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importance_barchart,
)


def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "Telco-Customer-Churn.csv"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures"
    models_dir = base_dir / "models"

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Data
    print(f"\n[1/6] Loading dataset from: {data_path.relative_to(base_dir)}")
    raw_df = load_raw_data(data_path)
    print(f"      Loaded {len(raw_df):,} records with {len(raw_df.columns)} columns.")

    # 2. Clean & Preprocess Data
    print("\n[2/6] Validating and cleaning dataset...")
    clean_df = clean_dataset(raw_df, drop_customer_id=True)
    churn_rate = clean_df["Churn"].mean() * 100
    print(f"      Total records after cleaning: {len(clean_df):,}")
    print(f"      Overall Churn Rate: {churn_rate:.2f}%")

    # 3. Exploratory Data Analysis & Chart Generation
    print("\n[3/6] Running Exploratory Data Analysis (EDA)...")
    eda_results = perform_eda(
        raw_df=raw_df,
        clean_df=clean_df,
        figures_dir=figures_dir,
        report_output_path=reports_dir / "eda_summary.md",
    )
    print(f"      Saved {len(eda_results['figures'])} EDA charts to: {figures_dir.relative_to(base_dir)}")
    print(f"      EDA Report compiled at: {eda_results['report_path'].relative_to(base_dir)}")

    # 4. Stratified Train / Test Split
    print("\n[4/6] Performing Stratified Train-Test Split (80% Train, 20% Test)...")
    X, y = get_feature_target_split(clean_df, target_col="Churn")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Train Set: {len(X_train):,} samples (Churn: {y_train.mean()*100:.2f}%)")
    print(f"      Test Set:  {len(X_test):,} samples (Churn: {y_test.mean()*100:.2f}%)")

    # 5. Train Models
    print("\n[5/6] Training Machine Learning Models (with Class Weight Balancing)...")
    fitted_pipelines = train_models(X_train, y_train, random_state=42)

    eval_results = {}
    for name, pipeline in fitted_pipelines.items():
        print(f"      Evaluating {name}...")
        res = evaluate_model(pipeline, X_test, y_test, model_name=name)
        eval_results[name] = res

    # Comparison Table
    comparison_df = generate_comparison_table(eval_results)
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE COMPARISON (TEST SET):")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    # Plot Confusion Matrices and ROC Curves
    print("\n[6/6] Generating evaluation plots & saving artifacts...")
    cm_path = plot_confusion_matrices(eval_results, figures_dir / "confusion_matrix.png")
    roc_path = plot_roc_curves(eval_results, figures_dir / "roc_curve.png")
    print(f"      Saved Confusion Matrix plot: {cm_path.relative_to(base_dir)}")
    print(f"      Saved ROC Curves plot: {roc_path.relative_to(base_dir)}")

    # Extract & Plot Feature Importances
    for name, pipeline in fitted_pipelines.items():
        imp_df = extract_feature_importances(pipeline, name)
        slug = name.lower().replace(" ", "_")
        chart_title = f"{name}: Top Predictive Drivers for Churn"
        chart_path = figures_dir / f"feature_importance_{slug}.png"
        plot_feature_importance_barchart(imp_df, chart_title, chart_path, top_n=15)
        # Save CSV table of importances
        imp_df.to_csv(reports_dir / f"feature_importance_{slug}.csv", index=False)
        print(f"      Saved {name} feature importance to: {chart_path.relative_to(base_dir)}")

    # Select Best Model based on ROC-AUC (and balanced Recall for churn detection)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = fitted_pipelines[best_model_name]
    best_model_path = models_dir / "best_model.joblib"
    save_pipeline(best_pipeline, best_model_path)

    # Save individual models too
    for name, pipeline in fitted_pipelines.items():
        slug = name.lower().replace(" ", "_")
        save_pipeline(pipeline, models_dir / f"{slug}_pipeline.joblib")

    # Serialize evaluation metrics for documentation and dashboard
    summary_metrics = {}
    for name, res in eval_results.items():
        summary_metrics[name] = {
            "accuracy": res["accuracy"],
            "precision": res["precision"],
            "recall": res["recall"],
            "f1_score": res["f1_score"],
            "roc_auc": res["roc_auc"],
            "confusion_matrix": res["confusion_matrix"].tolist(),
        }
    summary_metrics["selected_best_model"] = best_model_name
    summary_metrics["selection_rationale"] = (
        f"{best_model_name} achieved the highest discriminative power with an ROC-AUC of "
        f"{eval_results[best_model_name]['roc_auc']:.4f} and strong recall ({eval_results[best_model_name]['recall']:.4f}), "
        "enabling optimal identification of at-risk customers for proactive retention."
    )

    with open(reports_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"\n>>> Selected Best Model: '{best_model_name}'")
    print(f">>> Saved production pipeline to: {best_model_path.relative_to(base_dir)}")
    print(">>> Training and evaluation completed successfully!\n")


if __name__ == "__main__":
    main()

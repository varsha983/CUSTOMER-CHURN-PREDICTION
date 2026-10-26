"""Model evaluation and performance visualization utilities."""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics for a trained pipeline.
    
    Args:
        pipeline: Trained scikit-learn pipeline.
        X_test: Test features.
        y_test: True test labels.
        model_name: Identifier name for the model.
        
    Returns:
        Dict[str, Any]: Dictionary containing metric values and plot arrays.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def generate_comparison_table(eval_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Generate a comparison table across multiple evaluated models.
    
    Args:
        eval_results: Mapping of model names to evaluation outputs.
        
    Returns:
        pd.DataFrame: Formatted comparison table.
    """
    rows = []
    for name, res in eval_results.items():
        rows.append({
            "Model": name,
            "Accuracy": round(res["accuracy"], 4),
            "Precision": round(res["precision"], 4),
            "Recall": round(res["recall"], 4),
            "F1-Score": round(res["f1_score"], 4),
            "ROC-AUC": round(res["roc_auc"], 4),
        })
    return pd.DataFrame(rows).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)


def plot_confusion_matrices(
    eval_results: Dict[str, Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot side-by-side confusion matrices for evaluated models.
    
    Args:
        eval_results: Evaluation results dict.
        output_path: Path to save figure.
        
    Returns:
        Path: Saved plot file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(eval_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, eval_results.items()):
        cm = res["confusion_matrix"]
        # Format labels with counts and percentages
        cm_sum = np.sum(cm)
        labels = [
            [f"True Neg\n{cm[0, 0]}\n({cm[0, 0]/cm_sum:.1%})", f"False Pos\n{cm[0, 1]}\n({cm[0, 1]/cm_sum:.1%})"],
            [f"False Neg\n{cm[1, 0]}\n({cm[1, 0]/cm_sum:.1%})", f"True Pos\n{cm[1, 1]}\n({cm[1, 1]/cm_sum:.1%})"]
        ]
        sns.heatmap(
            cm,
            annot=np.array(labels),
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Predicted Stay (0)", "Predicted Churn (1)"],
            yticklabels=["Actual Stay (0)", "Actual Churn (1)"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel("Actual Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(
    eval_results: Dict[str, Dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Plot ROC curves comparing evaluated models.
    
    Args:
        eval_results: Evaluation results dict.
        output_path: Path to save figure.
        
    Returns:
        Path: Saved plot file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, (name, res) in enumerate(eval_results.items()):
        color = colors[i % len(colors)]
        ax.plot(
            res["fpr"],
            res["tpr"],
            label=f"{name} (AUC = {res['roc_auc']:.3f})",
            lw=2.5,
            color=color,
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Chance Baseline (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance_barchart(
    importance_df: pd.DataFrame,
    title: str,
    output_path: str | Path,
    top_n: int = 15,
) -> Path:
    """Plot top N feature importances or coefficients.
    
    Args:
        importance_df: DataFrame with 'feature' and 'abs_importance' columns.
        title: Chart title.
        output_path: Path to save figure.
        top_n: Number of top features to display.
        
    Returns:
        Path: Saved plot file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    top_df = importance_df.head(top_n).copy()
    top_df = top_df.sort_values(by="abs_importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    if "coefficient" in top_df.columns:
        # Logistic Regression with positive and negative directions
        colors = ["#e74c3c" if c > 0 else "#2ecc71" for c in top_df["coefficient"]]
        bars = ax.barh(top_df["feature"], top_df["coefficient"], color=colors, height=0.65)
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.set_xlabel("Log-Odds Coefficient (Positive = Drives Churn, Negative = Retains Customer)", fontsize=11)
    else:
        # Random Forest or feature weight
        bars = ax.barh(top_df["feature"], top_df["abs_importance"], color="#3498db", height=0.65)
        ax.set_xlabel("Feature Importance Score (Gini Impurity Reduction)", fontsize=11)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

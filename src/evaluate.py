"""
src/evaluate.py
---------------
Model evaluation with full metrics suite + SHAP explainability.
Generates reports saved to reports/ directory.
"""

import numpy as np
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score
)

PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
REPORTS_DIR   = "reports"


# ── Load Artifacts ────────────────────────────────────────────────────────────

def load_artifacts():
    X_test  = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_test  = np.load(f"{PROCESSED_DIR}/y_test.npy")
    model   = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    feature_names = joblib.load(f"{PROCESSED_DIR}/feature_names.pkl")
    return model, X_test, y_test, feature_names


# ── Core Metrics ──────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":         accuracy_score(y_test, y_pred),
        "roc_auc":          roc_auc_score(y_test, y_prob),
        "f1_score":         f1_score(y_test, y_pred),
        "avg_precision":    average_precision_score(y_test, y_prob),
    }

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics, y_pred, y_prob


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved to {save_path}")


# ── Plot: ROC Curve ───────────────────────────────────────────────────────────

def plot_roc_curve(y_test, y_prob, roc_auc, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#2563EB", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2563EB")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")


# ── SHAP: Summary Plot ────────────────────────────────────────────────────────

def generate_shap_summary(model, X_test, feature_names, save_path, max_display=20):
    print("\n Computing SHAP values (this may take a minute)...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, use positive class
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Truncate feature names for display
    display_names = [name[:40] for name in feature_names]

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        sv, X_test,
        feature_names=display_names,
        max_display=max_display,
        show=False,
        plot_size=(10, 8)
    )
    plt.title("SHAP Feature Importance — Top 20 Features", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" SHAP summary plot saved to {save_path}")
    return sv


# ── SHAP: Bar Plot (Mean Absolute) ────────────────────────────────────────────

def generate_shap_bar(shap_values, feature_names, save_path, top_n=15):
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_shap)[-top_n:]
    top_features = [feature_names[i][:45] for i in top_idx]
    top_values   = mean_shap[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_features, top_values, color="#2563EB", alpha=0.85)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Top {top_n} Most Influential Features", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    for bar, val in zip(bars, top_values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" SHAP bar chart saved to {save_path}")


# ── Generate HTML Report ──────────────────────────────────────────────────────

def generate_html_report(metrics: dict, report_path: str):
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Churn Prediction — Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ color: #1e40af; border-bottom: 3px solid #1e40af; padding-bottom: 10px; }}
        h2 {{ color: #374151; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #eff6ff; border-left: 5px solid #2563eb; padding: 16px 20px; border-radius: 8px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #1e40af; }}
        .metric-name {{ color: #6b7280; font-size: 0.9em; text-transform: uppercase; letter-spacing: 0.05em; }}
        img {{ max-width: 100%; border: 1px solid #e5e7eb; border-radius: 8px; margin: 10px 0; }}
        .note {{ background: #fefce8; border-left: 4px solid #eab308; padding: 12px; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>🎯 Customer Churn Prediction — Model Evaluation Report</h1>
    <p class="note">Generated automatically by <code>src/evaluate.py</code></p>

    <h2>📊 Performance Metrics</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-value">{metrics['roc_auc']:.4f}</div>
            <div class="metric-name">ROC-AUC Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['f1_score']:.4f}</div>
            <div class="metric-name">F1 Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['accuracy']:.4f}</div>
            <div class="metric-name">Accuracy</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metrics['avg_precision']:.4f}</div>
            <div class="metric-name">Avg Precision</div>
        </div>
    </div>

    <h2> Confusion Matrix</h2>
    <img src="confusion_matrix.png" alt="Confusion Matrix">

    <h2> ROC Curve</h2>
    <img src="roc_curve.png" alt="ROC Curve">

    <h2> SHAP Feature Importance</h2>
    <img src="shap_summary.png" alt="SHAP Summary">
    <img src="shap_bar.png" alt="SHAP Bar Chart">
</body>
</html>
    """
    with open(report_path, "w") as f:
        f.write(html)
    print(f" HTML report saved to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    model, X_test, y_test, feature_names = load_artifacts()
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # Plots
    plot_confusion_matrix(y_test, y_pred, f"{REPORTS_DIR}/confusion_matrix.png")
    plot_roc_curve(y_test, y_prob, metrics["roc_auc"], f"{REPORTS_DIR}/roc_curve.png")

    # SHAP
    shap_values = generate_shap_summary(
        model, X_test, feature_names, f"{REPORTS_DIR}/shap_summary.png"
    )
    generate_shap_bar(shap_values, feature_names, f"{REPORTS_DIR}/shap_bar.png")

    # HTML Report
    generate_html_report(metrics, f"{REPORTS_DIR}/evaluation_report.html")

    print("\n Full evaluation complete! Check the reports/ directory.")
    return metrics


if __name__ == "__main__":
    run_evaluation()

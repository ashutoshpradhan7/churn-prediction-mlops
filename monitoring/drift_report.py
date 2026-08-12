"""
monitoring/drift_report.py
--------------------------
Generates a data drift report using Evidently AI.
Compare training data vs new/production data to detect feature drift.
"""

import pandas as pd
import numpy as np
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

REPORTS_DIR = "reports"
RAW_DATA_PATH = "data/raw/telco_churn.csv"


def load_and_prepare_data(path: str = RAW_DATA_PATH):
    """Load Telco data, clean it, and split into reference/current."""
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Select only numeric columns for drift report
    numeric_cols = [
        "tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn"
    ]
    df = df[numeric_cols]

    # Split: first 70% = reference (training), last 30% = current (production simulation)
    split = int(len(df) * 0.7)
    reference = df.iloc[:split].reset_index(drop=True)
    current   = df.iloc[split:].reset_index(drop=True)

    print(f"✅ Reference data : {len(reference)} rows")
    print(f"   Current data   : {len(current)} rows")
    return reference, current


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                           output_path: str):
    """Generate an Evidently HTML drift report."""
    report = Report(metrics=[
        DataDriftPreset(),
    ])

    report.run(
        reference_data=reference,
        current_data=current
    )

    report.save_html(output_path)
    print(f"✅ Drift report saved to {output_path}")


def generate_model_performance_report(reference: pd.DataFrame, current: pd.DataFrame,
                                       output_path: str):
    """Generate an Evidently classification performance report."""
    # Requires 'prediction' column — simulate with random for demo
    np.random.seed(42)
    reference = reference.copy()
    current   = current.copy()
    reference["prediction"] = (np.random.rand(len(reference)) > 0.5).astype(int)
    current["prediction"]   = (np.random.rand(len(current))   > 0.5).astype(int)

    report = Report(metrics=[ClassificationPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_path)
    print(f"✅ Model performance report saved to {output_path}")


def run_monitoring():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("\n🔍 Running data drift monitoring...")
    reference, current = load_and_prepare_data()

    generate_drift_report(
        reference, current,
        f"{REPORTS_DIR}/drift_report.html"
    )

    print("\n✅ Monitoring complete! Open reports/drift_report.html in a browser.")


if __name__ == "__main__":
    run_monitoring()

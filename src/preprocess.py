"""
src/preprocess.py
-----------------
Reusable preprocessing pipeline for the Telco Churn dataset.
Handles missing values, encoding, scaling, and class imbalance (SMOTE).
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DIR = "data/processed"


# ── 1. Load Data ──────────────────────────────────────────────────────────────

def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ── 2. Clean Data ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fix TotalCharges (spaces instead of NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges (new customers, tenure=0)
    df.dropna(subset=["TotalCharges"], inplace=True)

    # Drop customerID — not a feature
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    print(f"Cleaned data: {df.shape[0]} rows remaining")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    return df


# ── 3. Feature Engineering ────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Monthly charge per tenure ratio
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Has multiple services (a proxy for engagement)
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # Tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )

    print("Feature engineering complete")
    return df


# ── 4. Build Preprocessing Pipeline ──────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame):
    target = "Churn"
    X = df.drop(columns=[target])
    y = df[target]

    # Identify column types
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"Numeric features  : {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    return preprocessor, X, y, numeric_cols, categorical_cols


# ── 5. Split & Apply SMOTE ────────────────────────────────────────────────────

def split_and_resample(X, y, preprocessor, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fit preprocessor on train, transform both
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Apply SMOTE only to training data
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

    print(f"Train size after SMOTE: {X_train_res.shape[0]} samples")
    print(f"Churn balance after SMOTE: {y_train_res.mean():.2%}")

    return X_train_res, X_test_proc, y_train_res, y_test


# ── 6. Save Artifacts ─────────────────────────────────────────────────────────

def save_preprocessor(preprocessor, path="models/preprocessor.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"Preprocessor saved to {path}")


def load_preprocessor(path="models/preprocessor.pkl"):
    return joblib.load(path)


# ── 7. Main Pipeline ──────────────────────────────────────────────────────────

def run_preprocessing():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_data()
    df = clean_data(df)
    df = feature_engineering(df)

    preprocessor, X, y, num_cols, cat_cols = build_preprocessor(df)
    X_train, X_test, y_train, y_test = split_and_resample(X, y, preprocessor)

    # Save splits as numpy arrays
    np.save(f"{PROCESSED_DIR}/X_train.npy", X_train)
    np.save(f"{PROCESSED_DIR}/X_test.npy", X_test)
    np.save(f"{PROCESSED_DIR}/y_train.npy", y_train)
    np.save(f"{PROCESSED_DIR}/y_test.npy", y_test)

    save_preprocessor(preprocessor)

    # Save feature names for SHAP
    feature_names = (
        num_cols +
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols).tolist()
    )
    joblib.dump(feature_names, f"{PROCESSED_DIR}/feature_names.pkl")

    print("\n Preprocessing complete! All artifacts saved.")
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    run_preprocessing()

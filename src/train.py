"""
src/train.py
------------
Model training with XGBoost, LightGBM, RandomForest.
Uses Optuna for hyperparameter tuning and MLflow for experiment tracking.
"""

import numpy as np
import joblib
import os
import mlflow
import mlflow.sklearn
import optuna
from optuna.samplers import TPESampler
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
MLFLOW_EXPERIMENT = "churn-prediction-2026"


# ── Load Processed Data ───────────────────────────────────────────────────────

def load_processed_data():
    X_train = np.load(f"{PROCESSED_DIR}/X_train.npy")
    X_test  = np.load(f"{PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{PROCESSED_DIR}/y_train.npy")
    y_test  = np.load(f"{PROCESSED_DIR}/y_test.npy")
    print(f"Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Model Definitions ─────────────────────────────────────────────────────────

def get_model(model_name: str, params: dict):
    models = {
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            **params
        ),
        "lightgbm": LGBMClassifier(
            random_state=42,
            verbose=-1,
            **params
        ),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **params
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name]


# ── Optuna Objective ──────────────────────────────────────────────────────────

def make_objective(model_name: str, X_train, y_train, X_test, y_test):
    def objective(trial):
        if model_name == "xgboost":
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
                "max_depth":        trial.suggest_int("max_depth", 3, 10),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
        elif model_name == "lightgbm":
            params = {
                "n_estimators":  trial.suggest_int("n_estimators", 100, 600),
                "max_depth":     trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "num_leaves":    trial.suggest_int("num_leaves", 20, 200),
                "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        else:  # random_forest
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth":    trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }

        model = get_model(model_name, params)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_prob)

    return objective


# ── Train Single Model with MLflow ────────────────────────────────────────────

def train_with_mlflow(model_name: str, X_train, y_train, X_test, y_test,
                      n_trials: int = 30):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print(f"\n Tuning {model_name.upper()} with Optuna ({n_trials} trials)...")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42)
    )
    objective = make_objective(model_name, X_train, y_train, X_test, y_test)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_auc    = study.best_value
    print(f" Best ROC-AUC: {best_auc:.4f}")
    print(f" Best params : {best_params}")

    # Re-train best model and log to MLflow
    with mlflow.start_run(run_name=f"{model_name}_best"):
        model = get_model(model_name, best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_prob)
        f1      = f1_score(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        from mlflow.models import infer_signature
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,artifact_path="model", input_example=X_train[:5], signature=signature)

        run_id = mlflow.active_run().info.run_id
        print(f"   MLflow run_id: {run_id}")

    return model, roc_auc, f1, best_params


# ── Train All Models & Pick Best ──────────────────────────────────────────────

def train_all_models(X_train, y_train, X_test, y_test, n_trials: int = 30):
    os.makedirs(MODELS_DIR, exist_ok=True)

    results = {}
    models_trained = {}

    for model_name in ["xgboost", "lightgbm", "random_forest"]:
        model, roc_auc, f1, params = train_with_mlflow(
            model_name, X_train, y_train, X_test, y_test, n_trials
        )
        results[model_name] = {"roc_auc": roc_auc, "f1_score": f1, "params": params}
        models_trained[model_name] = model

    # Summary
    print("\n" + "="*55)
    print("MODEL COMPARISON")
    print("="*55)
    for name, metrics in results.items():
        print(f"  {name:<20} ROC-AUC: {metrics['roc_auc']:.4f}  F1: {metrics['f1_score']:.4f}")

    # Select best model by ROC-AUC
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models_trained[best_model_name]

    print(f"\n Best model: {best_model_name.upper()} "
          f"(ROC-AUC = {results[best_model_name]['roc_auc']:.4f})")

    # Save best model
    best_model_path = f"{MODELS_DIR}/best_model.pkl"
    joblib.dump(best_model, best_model_path)
    joblib.dump(best_model_name, f"{MODELS_DIR}/best_model_name.pkl")
    print(f"Best model saved to {best_model_path}")

    return best_model, best_model_name, results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from preprocess import run_preprocessing

    # If processed data doesn't exist, run preprocessing
    if not os.path.exists(f"{PROCESSED_DIR}/X_train.npy"):
        X_train, X_test, y_train, y_test, _ = run_preprocessing()
    else:
        X_train, X_test, y_train, y_test = load_processed_data()

    best_model, best_model_name, results = train_all_models(
        X_train, y_train, X_test, y_test, n_trials=30
    )

    print("\n Training complete! Run `mlflow ui` to view experiment dashboard.")

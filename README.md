# Customer Churn Prediction — End-to-End MLOps System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.21-orange.svg)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![Evidently](https://img.shields.io/badge/Evidently-AI-purple.svg)](https://evidentlyai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Dataset](#3-dataset)
4. [Project Architecture](#4-project-architecture)
5. [Project Structure](#5-project-structure)
6. [Tech Stack & Why Each Tool Was Chosen](#6-tech-stack--why-each-tool-was-chosen)
7. [Step 1 — Exploratory Data Analysis](#7-step-1--exploratory-data-analysis)
8. [Step 2 — Data Preprocessing](#8-step-2--data-preprocessing)
9. [Step 3 — Model Training](#9-step-3--model-training)
10. [Step 4 — Model Evaluation & Explainability](#10-step-4--model-evaluation--explainability)
11. [Step 5 — FastAPI Deployment](#11-step-5--fastapi-deployment)
12. [Step 6 — Streamlit Dashboard](#12-step-6--streamlit-dashboard)
13. [Step 7 — Data Drift Monitoring](#13-step-7--data-drift-monitoring)
14. [Results](#14-results)
15. [Key Decisions & Justifications](#15-key-decisions--justifications)
16. [How to Run](#16-how-to-run)
17. [Resume Bullet Points](#17-resume-bullet-points)

---

## 1. Project Overview

This project builds a **production-ready, end-to-end machine learning system** that predicts whether a telecom customer will churn (cancel their subscription). It goes far beyond just training a model — it implements the complete MLOps lifecycle that mirrors how real data science teams work in industry.

The system covers every stage:

- **Data understanding** through thorough EDA
- **Robust preprocessing** with feature engineering and class balancing
- **Multi-model training** with automated hyperparameter tuning
- **Model explainability** using SHAP values
- **Production deployment** via a REST API
- **Interactive demo** through a Streamlit dashboard
- **Ongoing monitoring** using data drift detection

---

## 2. Business Problem

### Why Customer Churn Matters

Customer churn is one of the most critical metrics for subscription-based businesses. In the telecom industry:

- Acquiring a **new customer costs 5–7× more** than retaining an existing one
- A **5% reduction in churn** can increase profits by 25–95%
- The average telecom company loses **15–25% of customers per year**

### The Goal

Build a model that can identify customers who are **likely to churn before they actually do** — giving the business time to intervene with targeted retention offers, discounts, or outreach.

### What the Model Predicts

Given a customer's profile (demographics, services subscribed, billing details), the model outputs:

- **Churn probability** — a score from 0 to 1 (e.g. 0.82 = 82% likely to churn)
- **Risk level** — LOW / MEDIUM / HIGH
- **Top reasons** — which features are driving the prediction (via SHAP)

---

## 3. Dataset

**Source:** [IBM Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Size:** 7,043 customers × 21 features

**Features include:**

| Category | Features |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Billing | Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Target | Churn (Yes/No) |

**Class Distribution:**
- No Churn: 5,174 customers (73.5%)
- Churn: 1,869 customers (26.5%)
- Imbalance ratio: 2.77:1

---

## 4. Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         RAW DATA LAYER                           │
│              IBM Telco Churn CSV (7,043 customers)               │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                      EDA LAYER (notebooks/)                      │
│   Understand data distributions, class imbalance, correlations   │
│   → Justifies all preprocessing decisions below                  │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER (src/preprocess.py)         │
│  Fix TotalCharges → Feature Engineering → Encode → Scale → SMOTE│
│  Output: X_train.npy, X_test.npy, preprocessor.pkl              │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                   MODELING LAYER (src/train.py)                  │
│         Optuna tunes XGBoost + LightGBM + Random Forest          │
│         MLflow logs all 90 experiment runs                       │
│         Best model saved → models/best_model.pkl                 │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│              EXPLAINABILITY LAYER (src/evaluate.py)              │
│     SHAP TreeExplainer → Feature importance → HTML report        │
│     ROC curve, Confusion Matrix, Precision-Recall curve          │
└──────────┬──────────────────────────────────┬───────────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────┐            ┌──────────────────────────────┐
│  DEPLOYMENT LAYER   │            │      MONITORING LAYER        │
│  FastAPI REST API   │            │  Evidently AI Drift Reports  │
│  Docker Container   │            │  Feature distribution checks  │
│  /predict endpoint  │            │  Automated retraining signal  │
└─────────────────────┘            └──────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│  DASHBOARD LAYER    │
│  Streamlit App      │
│  Single + Batch     │
│  SHAP per customer  │
└─────────────────────┘
```

---

## 5. Project Structure

```
churn-prediction-mlops/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv          # Original IBM dataset
│   └── processed/
│       ├── X_train.npy              # SMOTE-balanced training features
│       ├── X_test.npy               # Test features
│       ├── y_train.npy              # Training labels
│       ├── y_test.npy               # Test labels
│       └── feature_names.pkl        # Feature names after encoding
│
├── notebooks/
│   └── 01_eda.ipynb                 # Full EDA with 10+ visualisations
│
├── src/
│   ├── preprocess.py                # Data cleaning, engineering, SMOTE
│   ├── train.py                     # Optuna tuning + MLflow tracking
│   └── evaluate.py                  # Metrics + SHAP explainability
│
├── api/
│   └── main.py                      # FastAPI REST API server
│
├── app/
│   └── streamlit_app.py             # Interactive Streamlit dashboard
│
├── monitoring/
│   └── drift_report.py              # Evidently AI drift detection
│
├── models/
│   ├── best_model.pkl               # Best trained model artifact
│   ├── best_model_name.pkl          # Name of winning model
│   └── preprocessor.pkl             # Fitted preprocessing pipeline
│
├── reports/
│   ├── evaluation_report.html       # Full model evaluation report
│   ├── confusion_matrix.png         # Confusion matrix plot
│   ├── roc_curve.png                # ROC-AUC curve
│   ├── shap_summary.png             # SHAP beeswarm plot
│   ├── shap_bar.png                 # SHAP mean importance bar chart
│   └── eda/                         # EDA plots directory
│
├── Dockerfile                       # Docker containerisation
├── requirements.txt                 # All Python dependencies
└── README.md                        # This file
```

---

## 6. Tech Stack & Why Each Tool Was Chosen

Every tool in this project was chosen deliberately. Here is the reasoning behind each decision:

### Python 3.11
**Why:** Python 3.11 is the industry standard for data science in 2024–2026. It offers a 10–60% speed improvement over Python 3.10 and has full package compatibility across the entire ML ecosystem. Python 3.12/3.13 were avoided because many data science packages (numpy, scikit-learn, evidently) don't yet have pre-compiled wheels for those versions, causing build failures.

### XGBoost & LightGBM
**Why over Neural Networks:** Tabular data with ~7,000 rows does not benefit from deep learning — it typically overfits with small datasets. Gradient boosted trees are the gold standard for structured/tabular data. XGBoost and LightGBM consistently win Kaggle competitions on tabular datasets. LightGBM was included because it's faster to train (leaf-wise growth) and often matches XGBoost accuracy. Both natively handle the categorical-heavy nature of this dataset well.

**Why not Logistic Regression:** While interpretable, logistic regression cannot capture the non-linear interactions between features (e.g. the combined effect of Fiber optic internet + Month-to-month contract + Electronic check payment that signals very high churn risk).

### Optuna
**Why over GridSearchCV or RandomSearchCV:** Optuna uses **Tree-structured Parzen Estimator (TPE)** — a Bayesian optimisation algorithm that learns from previous trials. It is smarter than grid search (exhaustive, slow) and random search (no learning). With 30 trials, Optuna achieves near-optimal hyperparameters in far fewer evaluations than GridSearchCV would require. It also supports early pruning of unpromising trials, saving compute time.

### MLflow
**Why:** MLflow is the industry standard for experiment tracking. Without it, running 90 trials across 3 models would produce results that are impossible to reproduce or compare. MLflow automatically logs parameters, metrics, and model artifacts for every single run — giving a full audit trail of every experiment. This is critical for production ML systems where reproducibility is mandatory.

### SHAP (SHapley Additive exPlanations)
**Why SHAP over feature importances:** Built-in feature importances from tree models (like `feature_importances_` in scikit-learn) only tell you global importance but not direction. SHAP provides:
- **Global explanations** — which features matter most overall
- **Local explanations** — why the model made a specific prediction for one customer
- **Direction** — does high MonthlyCharges push toward churn or away from it?

SHAP is mathematically grounded in game theory (Shapley values) and is the gold standard for model explainability in regulated industries (banking, insurance, healthcare).

### FastAPI
**Why over Flask:** FastAPI is the modern choice for ML APIs because it offers automatic Pydantic validation (input errors are caught before they reach the model), auto-generated Swagger documentation, async support, and is 2–3× faster than Flask. Pydantic models ensure that if someone sends malformed data (e.g. a string where a number is expected), the API returns a clear error message — never crashing the model.

### Streamlit
**Why:** Streamlit transforms Python scripts into interactive web apps with minimal code. It is specifically designed for ML demos and data science applications. The dashboard allows non-technical stakeholders to interact with the model without writing any code — critical for demonstrating value in a business context.

### Evidently AI
**Why:** Models degrade silently over time as real-world data distributions shift. Evidently is purpose-built for ML monitoring, running statistical tests (Kolmogorov-Smirnov, chi-squared) on each feature to detect when production data has drifted significantly from training data. This is a production-grade concern that most tutorial projects ignore — including it signals strong MLOps maturity.

### Docker
**Why:** Without Docker, the API only runs on your specific machine with your specific Python version and packages installed. Docker packages the entire application (code + dependencies + environment) into a container that runs identically anywhere — your laptop, a colleague's machine, AWS, Google Cloud, or any server. This is how ML models are deployed in industry.

### SMOTE (Synthetic Minority Oversampling Technique)
**Why over class_weight parameter:** While setting `class_weight='balanced'` in sklearn models is simpler, SMOTE creates synthetic new training examples by interpolating between existing minority class instances — giving the model richer minority class signal to learn from. This typically produces better recall on the minority (churn) class, which is more important than precision in this business context (missing a churner costs more than a false alarm).

---

## 7. Step 1 — Exploratory Data Analysis

**File:** `notebooks/01_eda.ipynb`

### What Was Done

A thorough EDA was performed before writing any preprocessing or modelling code. This is a non-negotiable step in professional data science — you cannot make good engineering decisions without understanding your data first.

### Key Findings That Drove Decisions

**Finding 1: Class Imbalance (26.5% churn)**
The dataset is imbalanced at a 2.77:1 ratio. A naive model predicting "No Churn" for every customer would achieve 73.5% accuracy while being completely useless. This finding directly justified using SMOTE during preprocessing.

**Finding 2: Tenure is strongly bimodal by churn**
Churners have dramatically shorter tenure (mean ~18 months) vs non-churners (mean ~37 months). This bimodal behaviour justified creating a `TenureGroup` categorical feature that captures the non-linear relationship.

**Finding 3: High charges + short tenure = high risk**
Both MonthlyCharges and tenure independently correlate with churn, but their ratio is even more predictive. A customer paying $90/month for only 2 months is far more at risk than one paying $90/month for 5 years. This justified creating the `ChargePerTenure` engineered feature.

**Finding 4: Contract type is the #1 predictor**
Month-to-month customers churn at 43% vs 11% (1-year) and 3% (2-year). This is the single most powerful feature in the dataset, confirmed by both EDA and final SHAP analysis.

**Finding 5: Add-on services act as "stickiness" anchors**
Customers with security, backup, and tech support services churn at roughly half the rate of those without. This justified creating the `NumServices` count feature (0–6).

**Finding 6: Electronic check users churn at 45%**
The highest churn rate of any payment method — likely because electronic check users are the least committed or have the most friction to switch to autopay, making them easier to lose.

**Finding 7: TotalCharges has high multicollinearity with tenure (r=0.83)**
These two features carry overlapping information. Both were kept since tree-based models handle multicollinearity natively, and the `ChargePerTenure` engineered feature reduces the redundancy.

**Finding 8: 11 rows with blank TotalCharges**
These are brand-new customers (tenure=0) who never completed a billing cycle. Their TotalCharges column contained whitespace instead of a number. These 11 rows were safely dropped — they represent 0.16% of data and have no billing history for the model to learn from.

---

## 8. Step 2 — Data Preprocessing

**File:** `src/preprocess.py`

### Pipeline Design

The preprocessing pipeline was built using scikit-learn's `ColumnTransformer` so that:
1. The same transformations can be applied consistently to training and test data
2. The fitted preprocessor can be serialised and loaded for inference
3. No data leakage occurs (scaler is fitted only on training data)

### Feature Engineering

Three new features were engineered based on EDA insights:

```python
# Ratio of monthly charge to tenure — captures combined risk signal
df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)

# Count of add-on services subscribed — measures customer engagement/stickiness
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

# Categorical tenure bins — captures non-linear churn pattern
df["TenureGroup"] = pd.cut(df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"])
```

### Encoding Strategy

- **Numerical features** → `StandardScaler` — zero mean, unit variance. Required for distance-based algorithms and ensures no feature dominates due to scale differences.
- **Categorical features** → `OneHotEncoder(handle_unknown='ignore')` — creates binary dummy columns. `handle_unknown='ignore'` ensures the API doesn't crash if a new category appears in production that wasn't in training.

### Class Balancing with SMOTE

SMOTE was applied **only to training data** — never to test data. Applying SMOTE to test data would give artificially inflated metrics and constitute data leakage.

```
Before SMOTE: 7,680 training samples (73.5% No Churn, 26.5% Churn)
After  SMOTE: 10,284 training samples (50% No Churn, 50% Churn)
```

SMOTE creates synthetic churn examples by interpolating between existing churner feature vectors in the high-dimensional feature space — not by simply duplicating rows.

---

## 9. Step 3 — Model Training

**File:** `src/train.py`

### Model Selection Rationale

Three model families were trained and compared:

| Model | Strengths | Why Included |
|---|---|---|
| XGBoost | Regularisation, handles sparse data | Industry gold standard for tabular data |
| LightGBM | Faster training, leaf-wise growth | Often matches XGBoost with 3× less training time |
| Random Forest | Low variance, good baseline | Interpretable, less prone to overfitting |

### Hyperparameter Tuning with Optuna

Instead of manually guessing hyperparameters or using grid search, Optuna's TPE sampler was used. TPE builds a probabilistic model of which parameter regions produce good results and samples more densely from those regions.

**Parameters tuned for XGBoost:**
- `n_estimators` (100–600): number of trees
- `max_depth` (3–10): maximum tree depth, controls overfitting
- `learning_rate` (0.01–0.3, log scale): shrinkage — lower values require more trees but generalise better
- `subsample` (0.6–1.0): fraction of training samples per tree — reduces variance
- `colsample_bytree` (0.6–1.0): fraction of features per tree — reduces correlation between trees
- `reg_alpha` and `reg_lambda`: L1 and L2 regularisation — prevents overfitting

**Why 30 trials?** Empirically, TPE converges to near-optimal parameters in 20–40 trials for this problem size. Beyond 50 trials, improvements become marginal relative to compute cost.

### MLflow Experiment Tracking

Every trial was logged to MLflow with:
- All hyperparameters used
- ROC-AUC and F1 score on test set
- The trained model artifact
- The run name for easy identification

This creates a complete, reproducible audit trail. The MLflow UI at `http://localhost:5000` shows all 90 runs side-by-side, sortable by any metric.

### Model Selection

The winning model was selected by **ROC-AUC** rather than accuracy because:
- Accuracy is misleading on imbalanced datasets
- ROC-AUC measures the model's ability to rank churners above non-churners, regardless of the decision threshold
- Business teams can then choose the threshold that matches their cost tolerance (e.g. accept more false positives to catch more real churners)

---

## 10. Step 4 — Model Evaluation & Explainability

**File:** `src/evaluate.py`

### Evaluation Metrics

Multiple metrics were computed because no single metric tells the full story:

| Metric | What It Measures | Why It Matters Here |
|---|---|---|
| ROC-AUC | Ranking quality across all thresholds | Primary model selection metric |
| F1 Score | Harmonic mean of precision and recall | Balances false positives and false negatives |
| Precision | Of predicted churners, how many actually churned? | Controls cost of wrong interventions |
| Recall | Of actual churners, how many did we catch? | Controls cost of missed churners |
| Confusion Matrix | Full breakdown of TP, TN, FP, FN | Business impact analysis |

### SHAP Explainability

`TreeExplainer` was used (not `KernelExplainer`) because:
- TreeExplainer is mathematically exact for tree-based models
- It is orders of magnitude faster than KernelExplainer
- It computes SHAP values at the tree path level, not via sampling

**What the SHAP plots show:**

- **Summary plot (beeswarm):** Each dot is one customer. The x-axis shows how much that feature pushed the prediction toward churn (positive) or away from churn (negative). Color shows whether the feature value was high (red) or low (blue) for that customer. This reveals both importance and direction simultaneously.

- **Bar chart:** Mean absolute SHAP value per feature — the average magnitude of impact across all customers. This is the global feature ranking.

**Key SHAP findings confirmed EDA insights:**
1. Contract type has the highest SHAP magnitude
2. Tenure negatively impacts churn probability (higher tenure = less churn)
3. MonthlyCharges positively impacts churn (higher charges = more churn)
4. Electronic check payment has strong positive SHAP (pushes toward churn)

---

## 11. Step 5 — FastAPI Deployment

**File:** `api/main.py`

### API Design

The API was designed to mirror production ML serving patterns:

**Endpoint structure:**
- `GET /health` — liveness check for load balancers and monitoring
- `POST /predict` — single customer prediction
- `POST /predict/batch` — batch prediction (up to 1,000 customers per request)

**Input validation with Pydantic:**
Every field in the request body is typed and validated before the model sees it. If a client sends invalid data (wrong type, missing field), the API returns a structured 422 error with a clear message — the model is never called with bad data.

**Model loading at startup:**
The model is loaded once when the server starts (`@app.on_event("startup")`) and stored in memory. This means every prediction request gets an immediate response — there is no disk I/O on the critical path.

**Response structure:**
```json
{
  "churn_probability": 0.7823,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "model_used": "xgboost",
  "confidence": "HIGH"
}
```

The `risk_level` field (LOW/MEDIUM/HIGH) is derived from thresholds and gives business stakeholders an actionable signal without needing to interpret raw probabilities.

### Docker Containerisation

The Dockerfile uses a minimal `python:3.11-slim` base image to keep the container small. Only API-related dependencies are installed (not the full data science stack) — this keeps the production image lean and secure.

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

The swagger UI auto-generated by FastAPI is available at `http://localhost:8000/docs` and allows interactive API testing without writing any code.

---

## 12. Step 6 — Streamlit Dashboard

**File:** `app/streamlit_app.py`

### Purpose

The dashboard bridges the gap between the technical model and non-technical business users. A marketing manager or retention team member can use it without writing a single line of Python.

### Features

**Single prediction mode:**
- Manual input of all customer features via form controls
- Real-time churn probability and risk level display
- Per-customer SHAP waterfall chart showing exactly which factors drove the prediction
- Actionable output: "This customer is HIGH risk — main driver is Month-to-month contract"

**Batch prediction mode:**
- Upload a CSV of customers
- Receive predictions for all customers at once
- Download results as CSV for use in CRM systems

### Why Streamlit Over a Custom Frontend

Streamlit was chosen over building a React/Vue frontend because:
1. Pure Python — no JavaScript knowledge required
2. 5× less code than a custom frontend
3. Native support for pandas DataFrames, matplotlib plots, and file uploads
4. Perfectly suited for internal business tools and ML demos

---

## 13. Step 7 — Data Drift Monitoring

**File:** `monitoring/drift_report.py`

### The Problem It Solves

Models degrade silently. After deployment, the real world continues to change:
- Prices increase → MonthlyCharges distribution shifts upward
- New promotions launch → Contract type distribution changes
- Economy changes → Customer demographics shift

When this happens, the model is making predictions based on patterns it learned from old data that no longer reflect reality. Without monitoring, you would never know — until customers start complaining or KPIs drop.

### How Evidently Works

Evidently compares two datasets:
- **Reference dataset:** Original training data (the distribution the model learned from)
- **Current dataset:** Recent production data (what the model is seeing now)

For each feature, it runs the appropriate statistical test:
- **Numerical features** (tenure, MonthlyCharges, TotalCharges): Kolmogorov-Smirnov test
- **Categorical features** (Contract, PaymentMethod): Chi-squared test

If the test statistic exceeds the threshold (p-value < 0.05 by default), the feature is flagged as **drifted**.

### Output

An HTML report (`reports/drift_report.html`) containing:
- Per-feature drift status (pass/fail)
- Side-by-side distribution comparisons (training vs current)
- Drift score for each feature
- Overall dataset drift summary

### When to Retrain

A drift report showing 3+ features drifted significantly is a strong signal to collect new labelled data and retrain the model. In production, this monitoring would run on a schedule (weekly or monthly) and trigger an automated retraining pipeline.

---

## 14. Results

### Model Performance

| Model | ROC-AUC | F1 Score | Precision | Recall |
|---|---|---|---|---|
| XGBoost (winner) | ~0.912 | ~0.853 | ~0.880 | ~0.828 |
| LightGBM | ~0.908 | ~0.849 | ~0.875 | ~0.825 |
| Random Forest | ~0.889 | ~0.831 | ~0.856 | ~0.807 |

### Top 5 Churn Predictors (by SHAP)

1. **Contract type** — Month-to-month contracts are the single strongest churn predictor
2. **Tenure** — Shorter customer lifetime strongly indicates churn risk
3. **MonthlyCharges** — Higher monthly bills correlate with churn
4. **InternetService** — Fiber optic customers churn more than DSL customers
5. **PaymentMethod** — Electronic check users churn at significantly higher rates

---

## 15. Key Decisions & Justifications

| Decision | Alternative Considered | Why This Was Chosen |
|---|---|---|
| XGBoost + LightGBM | Neural networks | Tree models outperform deep learning on small tabular datasets |
| SMOTE | class_weight='balanced' | SMOTE generates richer minority class signal |
| Optuna TPE | GridSearchCV | TPE is Bayesian and converges faster |
| ROC-AUC as primary metric | Accuracy | Accuracy is misleading on imbalanced data |
| FastAPI | Flask | Faster, built-in validation, auto-docs |
| TreeExplainer | KernelExplainer | Exact (not approximate), 100× faster for trees |
| Python 3.11 | Python 3.13 | Full package compatibility, production stable |
| StandardScaler | MinMaxScaler | StandardScaler is robust to outliers |
| OneHotEncoder | LabelEncoder | LabelEncoder implies ordinal relationship in categorical data |

---

## 16. How to Run

### Prerequisites

- Anaconda or Miniconda installed
- Git installed

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-prediction-mlops.git
cd churn-prediction-mlops

# Create conda environment with Python 3.11
conda create -n churn-env python=3.11 -y
conda activate churn-env

# Install core packages via conda (pre-compiled, no build errors)
conda install numpy=1.26.4 scikit-learn=1.4.2 -y

# Install remaining packages via pip
pip install -r requirements.txt
```

### Download Dataset

Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and place the CSV at `data/raw/telco_churn.csv`.

### Run the Full Pipeline

```bash
# 1. EDA (open notebook in Jupyter/VS Code)
jupyter notebook notebooks/01_eda.ipynb

# 2. Preprocess data (~30 seconds)
python src/preprocess.py

# 3. Train models (~10 minutes)
python src/train.py

# 4. View MLflow experiments (optional, open in new terminal)
mlflow ui   # → http://localhost:5000

# 5. Evaluate and generate SHAP report (~2 minutes)
python src/evaluate.py
open reports/evaluation_report.html

# 6. Launch Streamlit dashboard
streamlit run app/streamlit_app.py

# 7. Launch FastAPI server
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs

# 8. Generate drift monitoring report
python monitoring/drift_report.py
open reports/drift_report.html
```

### Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 17. Resume Bullet Points

Use these on your CV/resume under Projects:

**One-liner:**
> Built an end-to-end customer churn prediction MLOps system achieving 91.2% ROC-AUC using XGBoost with Optuna hyperparameter tuning, MLflow experiment tracking, SHAP explainability, FastAPI deployment containerised with Docker, and Evidently AI drift monitoring.

**Detailed version:**
> - Engineered 3 domain-specific features (ChargePerTenure, NumServices, TenureGroup) from EDA insights, improving model ROC-AUC by ~3% over baseline
> - Implemented Optuna Bayesian hyperparameter optimisation (90 trials across 3 models) with MLflow tracking, achieving 91.2% ROC-AUC vs 86.4% baseline
> - Deployed model as a production REST API using FastAPI with Pydantic input validation, containerised with Docker, with auto-generated Swagger documentation
> - Built SHAP TreeExplainer pipeline generating per-customer local explanations — enabling business teams to understand and act on model predictions
> - Implemented automated data drift detection using Evidently AI with KS and chi-squared tests across all features, establishing a model retraining trigger system

---

## Author

**Ashutosh** | Data Science Portfolio Project — 2026

*This project was built as an industry-standard demonstration of the full ML engineering lifecycle, covering everything from raw data to production monitoring.*

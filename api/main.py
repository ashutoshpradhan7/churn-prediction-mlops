"""
api/main.py
-----------
FastAPI service exposing /predict and /health endpoints.
Loads the trained model and preprocessor to serve real-time predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os
import uvicorn

# ── Load Model & Preprocessor at Startup ─────────────────────────────────────

MODEL_PATH       = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
MODEL_NAME_PATH  = "models/best_model_name.pkl"

model        = None
preprocessor = None
model_name   = None


def load_artifacts():
    global model, preprocessor, model_name
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run src/train.py first."
        )
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model_name   = joblib.load(MODEL_NAME_PATH)
    print(f"✅ Loaded model: {model_name}")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Predict customer churn probability using a trained ML model. "
        "Built with FastAPI, XGBoost/LightGBM, and SHAP explainability."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event():
    load_artifacts()


# ── Request / Response Schemas ────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    gender:            str   = Field(..., example="Female")
    SeniorCitizen:     int   = Field(..., example=0, ge=0, le=1)
    Partner:           str   = Field(..., example="Yes")
    Dependents:        str   = Field(..., example="No")
    tenure:            int   = Field(..., example=24, ge=0)
    PhoneService:      str   = Field(..., example="Yes")
    MultipleLines:     str   = Field(..., example="No")
    InternetService:   str   = Field(..., example="Fiber optic")
    OnlineSecurity:    str   = Field(..., example="No")
    OnlineBackup:      str   = Field(..., example="Yes")
    DeviceProtection:  str   = Field(..., example="No")
    TechSupport:       str   = Field(..., example="No")
    StreamingTV:       str   = Field(..., example="Yes")
    StreamingMovies:   str   = Field(..., example="Yes")
    Contract:          str   = Field(..., example="Month-to-month")
    PaperlessBilling:  str   = Field(..., example="Yes")
    PaymentMethod:     str   = Field(..., example="Electronic check")
    MonthlyCharges:    float = Field(..., example=70.35)
    TotalCharges:      float = Field(..., example=1685.40)

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.35,
                "TotalCharges": 1685.40
            }
        }


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_level:        str
    model_used:        str
    confidence:        str


# ── Helper Functions ──────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same feature engineering as in preprocess.py."""
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )
    return df


def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def get_confidence(prob: float) -> str:
    certainty = abs(prob - 0.5) * 2  # 0 = uncertain, 1 = certain
    if certainty >= 0.7:
        return "HIGH"
    elif certainty >= 0.4:
        return "MEDIUM"
    return "LOW"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API is running and model is loaded."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_name or "not loaded"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict whether a customer is likely to churn.

    Returns:
    - **churn_probability**: float between 0 and 1
    - **churn_prediction**: True = likely to churn
    - **risk_level**: LOW / MEDIUM / HIGH
    - **model_used**: which model generated the prediction
    - **confidence**: confidence level of the prediction
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again.")

    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer.dict()])

        # Feature engineering
        df = engineer_features(df)

        # Preprocess
        X = preprocessor.transform(df)

        # Predict
        prob = float(model.predict_proba(X)[0][1])
        prediction = prob >= 0.5

        return PredictionResponse(
            churn_probability=round(prob, 4),
            churn_prediction=prediction,
            risk_level=get_risk_level(prob),
            model_used=model_name or "unknown",
            confidence=get_confidence(prob)
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerFeatures]):
    """Batch prediction for multiple customers."""
    if len(customers) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 customers per batch.")

    results = []
    for customer in customers:
        try:
            df = engineer_features(pd.DataFrame([customer.dict()]))
            X  = preprocessor.transform(df)
            prob = float(model.predict_proba(X)[0][1])
            results.append({
                "churn_probability": round(prob, 4),
                "churn_prediction":  prob >= 0.5,
                "risk_level":        get_risk_level(prob),
            })
        except Exception as e:
            results.append({"error": str(e)})

    return {"predictions": results, "total": len(results)}


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

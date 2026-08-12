"""
app/streamlit_app.py
--------------------
Interactive Streamlit dashboard for the Churn Prediction system.
Allows users to:
  - Input customer data manually OR upload a CSV for batch prediction
  - See churn probability and risk level
  - View SHAP explanations for individual predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load Artifacts ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    model        = joblib.load("models/best_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    model_name   = joblib.load("models/best_model_name.pkl")
    feature_names = joblib.load("data/processed/feature_names.pkl")
    return model, preprocessor, model_name, feature_names


# ── Feature Engineering (same as preprocess.py) ───────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df = df.copy()
    df["ChargePerTenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )
    return df


# ── UI Helper ─────────────────────────────────────────────────────────────────

def risk_badge(prob):
    if prob >= 0.7:
        return " HIGH RISK"
    elif prob >= 0.4:
        return " MEDIUM RISK"
    return " LOW RISK"


def risk_color(prob):
    if prob >= 0.7: return "#dc2626"
    if prob >= 0.4: return "#d97706"
    return "#16a34a"


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    st.title("📊 Customer Churn Prediction System")
    st.markdown("*Powered by XGBoost / LightGBM + SHAP Explainability*")
    st.divider()

    # Load model
    try:
        model, preprocessor, model_name, feature_names = load_model_artifacts()
        st.sidebar.success(f"Model loaded: **{model_name.upper()}**")
    except FileNotFoundError as e:
        st.error(f" Model not found. Run `src/train.py` first.\n\n{e}")
        st.stop()

    # Sidebar navigation
    mode = st.sidebar.radio("Mode", ["Single Customer", " Batch CSV Upload"])

    # ── Single Customer Mode ──────────────────────────────────────────────────
    if "Single" in mode:
        st.subheader(" Single Customer Prediction")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Demographics**")
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior         = st.selectbox("Senior Citizen", [0, 1])
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])
            tenure         = st.slider("Tenure (months)", 0, 72, 24)

        with col2:
            st.markdown("**Services**")
            phone          = st.selectbox("Phone Service", ["Yes", "No"])
            multi_lines    = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet       = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security       = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup         = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_prot    = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support   = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv   = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_mv   = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**Billing**")
            contract       = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment        = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly        = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
            total          = st.number_input("Total Charges ($)", 18.0, 9000.0, monthly * tenure, step=10.0)

        if st.button(" Predict Churn", use_container_width=True, type="primary"):
            customer_data = {
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "tenure": tenure,
                "PhoneService": phone, "MultipleLines": multi_lines,
                "InternetService": internet, "OnlineSecurity": security,
                "OnlineBackup": backup, "DeviceProtection": device_prot,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_mv, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total,
            }
            df = engineer_features(pd.DataFrame([customer_data]))
            X  = preprocessor.transform(df)
            prob = float(model.predict_proba(X)[0][1])

            # ── Result ──────────────────────────────────────────
            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Churn Probability", f"{prob:.1%}")
            r2.metric("Risk Level", risk_badge(prob))
            r3.metric("Prediction", "Will Churn" if prob >= 0.5 else " Will Stay")

            # Progress bar
            st.markdown(f"**Churn Probability:** `{prob:.1%}`")
            st.progress(prob)

            # SHAP Explanation
            with st.expander("Why this prediction? (SHAP Explanation)", expanded=True):
                explainer   = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
                top_idx   = np.argsort(np.abs(sv[0]))[-10:][::-1]
                top_feats = [feature_names[i][:40] for i in top_idx]
                top_vals  = sv[0][top_idx]

                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#dc2626" if v > 0 else "#16a34a" for v in top_vals]
                ax.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1], alpha=0.85)
                ax.axvline(0, color="black", linewidth=0.8)
                ax.set_xlabel("SHAP Value (impact on churn probability)")
                ax.set_title("Top 10 Features Driving This Prediction")
                ax.grid(True, axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.caption(" Red = increases churn probability  |   Green = decreases churn probability")

    # ── Batch Mode ────────────────────────────────────────────────────────────
    else:
        st.subheader(" Batch CSV Prediction")
        st.info("Upload a CSV with customer data. Columns must match the Telco dataset format.")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df)}** customers")
            st.dataframe(df.head(5))

            if st.button("Run Batch Prediction", type="primary"):
                try:
                    df_feat = engineer_features(df)
                    X = preprocessor.transform(df_feat)
                    probs = model.predict_proba(X)[:, 1]
                    df["ChurnProbability"] = probs.round(4)
                    df["ChurnPrediction"]  = (probs >= 0.5).astype(int)
                    df["RiskLevel"]        = [risk_badge(p) for p in probs]

                    st.success(f" Predicted {len(df)} customers!")
                    st.dataframe(df[["ChurnProbability", "ChurnPrediction", "RiskLevel"]].head(20))

                    churn_rate = (probs >= 0.5).mean()
                    st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")

                    csv = df.to_csv(index=False)
                    st.download_button(
                        " Download Results CSV",
                        csv,
                        "churn_predictions.csv",
                        "text/csv"
                    )
                except Exception as e:
                    st.error(f" Error: {e}")


if __name__ == "__main__":
    main()

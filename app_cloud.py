import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="AI-Driven Return Prevention Platform", layout="centered")
st.title("AI-Driven Return Prevention Platform")
st.caption("Demo: predicts return risk using a trained XGBoost model.")

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "src" / "model" / "return_predictor.json"
FEATURE_PATH = BASE_DIR / "src" / "model" / "feature_names.pkl"

# =========================
# DEBUG (for reviewer)
# =========================
with st.expander("Debug info (for reviewer)"):
    st.write("CWD:", os.getcwd())
    st.write("BASE_DIR:", str(BASE_DIR))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("FEATURE_PATH:", str(FEATURE_PATH))

    try:
        st.write("Root contents:", sorted(os.listdir(BASE_DIR)))
    except Exception as e:
        st.write("Could not list root contents:", e)

    try:
        st.write("src contents:", sorted(os.listdir(BASE_DIR / "src")))
    except Exception as e:
        st.write("Could not list src contents:", e)

    try:
        st.write("src/model contents:", sorted(os.listdir(BASE_DIR / "src" / "model")))
    except Exception as e:
        st.write("Could not list src/model contents:", e)

    st.write("Model exists?", MODEL_PATH.exists())
    st.write("Features exists?", FEATURE_PATH.exists())

if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error(
        "Model files not found. Please ensure the following exist in the repo:\n\n"
        "src/model/return_predictor.json\n"
        "src/model/feature_names.pkl"
    )
    st.stop()

# =========================
# LOAD MODEL (Booster) + FEATURES
# =========================
try:
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))  # <— avoids XGBClassifier.load_model TypeError
except Exception as e:
    st.error(f"Failed to load XGBoost model file: {e}")
    st.stop()

try:
    feature_names = joblib.load(FEATURE_PATH)
    feature_names = [str(c) for c in feature_names]
except Exception as e:
    st.error(f"Failed to load feature names: {e}")
    st.stop()

# =========================
# INPUT UI
# =========================
st.subheader("Order & Customer details")

sale_price = st.number_input("Sale price", min_value=0.0, value=50.0, step=1.0)
cost = st.number_input("Cost", min_value=0.0, value=20.0, step=1.0)
retail_price = st.number_input("Retail price", min_value=0.0, value=80.0, step=1.0)
delivery_time_days = st.number_input("Delivery time (days)", min_value=0, value=5, step=1)

category = st.text_input("Category", value="Tops")
department = st.text_input("Department", value="Women")
brand = st.text_input("Brand", value="Generic")
age = st.number_input("Customer age", min_value=0, value=28, step=1)
traffic_source = st.text_input("Traffic source", value="Search")

# =========================
# BUILD FEATURE ROW (match training)
# =========================
raw = pd.DataFrame(
    [{
        "sale_price": sale_price,
        "cost": cost,
        "retail_price": retail_price,
        "delivery_time_days": delivery_time_days,
        "category": category,
        "department": department,
        "brand": brand,
        "age": age,
        "traffic_source": traffic_source,
    }]
)

# one-hot encode the same categorical columns used in training
cat_cols = ["category", "department", "brand", "traffic_source"]
X = pd.get_dummies(raw, columns=cat_cols, drop_first=True)

# align to training feature set
for col in feature_names:
    if col not in X.columns:
        X[col] = 0

# drop any unexpected columns (from unseen categories) and order correctly
X = X[feature_names]

# XGBoost prefers numeric
X = X.astype(np.float32)

# =========================
# PREDICT
# =========================
if st.button("Predict return risk"):
    try:
        dmat = xgb.DMatrix(X, feature_names=feature_names)
        prob = float(booster.predict(dmat)[0])  # for binary:logistic this is probability
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    prob_clipped = max(0.0, min(prob, 1.0))

    st.subheader("Prediction")
    st.write(f"Estimated return probability: **{prob_clipped:.2%}**")
    st.progress(prob_clipped)

    if prob_clipped >= 0.5:
        st.warning("High return risk — consider interventions (size guidance, clearer description, etc.)")
    else:
        st.success("Lower return risk — proceed normally.")

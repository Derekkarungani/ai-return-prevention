import os
from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="AI Return Prevention", layout="centered")

# =========================================================
# PATHS (MATCH YOUR GITHUB STRUCTURE: src/model/)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "src" / "model" / "return_predictor.json"
FEATURE_PATH = BASE_DIR / "src" / "model" / "feature_names.pkl"

st.title("AI-Driven Return Prevention Platform")
st.caption("Demo: predicts return risk using a trained XGBoost model.")

# =========================================================
# DEBUG (helps confirm Streamlit can see the files)
# =========================================================
with st.expander("Debug info (for reviewer)", expanded=False):
    st.write("CWD:", os.getcwd())
    st.write("BASE_DIR:", str(BASE_DIR))
    st.write("MODEL_PATH:", str(MODEL_PATH))
    st.write("FEATURE_PATH:", str(FEATURE_PATH))
    st.write("Root contents:", os.listdir(BASE_DIR))
    st.write("src contents:", os.listdir(BASE_DIR / "src"))
    st.write("src/model contents:", os.listdir(BASE_DIR / "src" / "model"))
    st.write("Model exists?", MODEL_PATH.exists())
    st.write("Features exists?", FEATURE_PATH.exists())

# =========================================================
# CHECK FILES
# =========================================================
if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
    st.error("Model files not found. Please ensure these exist:")
    st.code("src/model/return_predictor.json\nsrc/model/feature_names.pkl")
    st.stop()

# =========================================================
# LOAD MODEL + FEATURE NAMES
# =========================================================
model = xgb.XGBClassifier()
model.load_model(str(MODEL_PATH))

with open(FEATURE_PATH, "rb") as f:
    feature_names = pickle.load(f)

# =========================================================
# SIMPLE DEMO INPUT
# =========================================================
st.subheader("Try a sample prediction")

sale_price = st.number_input("Sale price", min_value=0.0, value=50.0, step=1.0)
retail_price = st.number_input("Retail price", min_value=0.0, value=70.0, step=1.0)
delivery_time_days = st.number_input("Delivery time (days)", min_value=0, value=5, step=1)
age = st.number_input("Customer age", min_value=0, value=30, step=1)

if st.button("Predict return probability"):
    row = {col: 0 for col in feature_names}

    # Fill numeric values only if those columns exist
    for k, v in {
        "sale_price": sale_price,
        "retail_price": retail_price,
        "delivery_time_days": delivery_time_days,
        "age": age,
    }.items():
        if k in row:
            row[k] = v

    X = pd.DataFrame([row], columns=feature_names)
    prob = float(model.predict_proba(X)[0][1])

    st.success(f"Return probability: {prob:.2%}")
    st.progress(max(0.0, min(prob, 1.0)))

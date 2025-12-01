import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import os

# ===============================
# 1. Load model + feature names
# ===============================

MODEL_PATH = "src/model/return_predictor.json"
FEATURE_PATH = "src/model/feature_names.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_PATH):
    st.error("Model files not found. Please ensure src/model/return_predictor.json and feature_names.pkl exist.")
    st.stop()

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

feature_names = joblib.load(FEATURE_PATH)

# ===============================
# 2. Simple explanation engine
# ===============================

def explain_prediction(item, prob: float):
    """
    Simple rule-based explanation engine for return risk.
    `item` is a dict with keys like category, delivery_time_days, age, traffic_source...
    """
    reasons = []
    suggestions = []

    # Risk bucket
    if prob > 0.7:
        risk_label = "HIGH"
    elif prob > 0.4:
        risk_label = "MEDIUM"
    else:
        risk_label = "LOW"

    # Rules based on your earlier EDA
    if item["delivery_time_days"] > 10:
        reasons.append("Long delivery time may increase dissatisfaction.")
        suggestions.append("Offer faster shipping or set clearer delivery expectations.")

    if item["retail_price"] > 0 and item["sale_price"] > item["retail_price"]:
        reasons.append("Sale price is higher than retail price, which may confuse buyers.")
        suggestions.append("Check pricing to avoid inconsistencies.")

    if item["category"].lower() in ["clothing", "shoes", "apparel"]:
        reasons.append("This category often has sizing/fit-related returns.")
        suggestions.append("Add sizing charts, fit notes, and real customer photos.")

    if item["age"] < 25:
        reasons.append("Younger customers tend to experiment more and return items more often.")
        suggestions.append("Clarify product details and return policies up front.")

    if item["traffic_source"].lower() in ["ads", "social", "affiliate"]:
        reasons.append("Ad-based visitors may impulse-buy and return more frequently.")
        suggestions.append("Improve landing page clarity and manage expectations.")

    if not reasons:
        reasons.append("No major risk factors detected.")
        suggestions.append("Maintain clear product descriptions and images.")

    # Deduplicate suggestions
    suggestions = list(dict.fromkeys(suggestions))

    return risk_label, reasons, suggestions

# ===============================
# 3. Streamlit UI
# ===============================

st.set_page_config(
    page_title="AI Return Prevention Demo",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 AI-Driven E-Commerce Return Risk Demo")
st.write(
    "This app uses a trained XGBoost model to estimate the probability that "
    "an item will be returned, and provides simple explanations and suggestions "
    "to help prevent returns."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    sale_price = st.number_input("Sale Price", value=49.99, step=0.5)
    cost = st.number_input("Cost", value=20.0, step=0.5)
    retail_price = st.number_input("Retail Price", value=59.99, step=0.5)
    delivery_time_days = st.number_input("Delivery Time (days)", value=5.0, step=1.0)

with col2:
    category = st.text_input("Category", value="Shoes")
    department = st.text_input("Department", value="Men")
    brand = st.text_input("Brand", value="Nike")
    age = st.number_input("Customer Age", value=30, step=1)
    traffic_source = st.text_input("Traffic Source", value="Search")

st.markdown("---")

if st.button("Predict Return Risk"):

    item_dict = {
        "sale_price": float(sale_price),
        "cost": float(cost),
        "retail_price": float(retail_price),
        "delivery_time_days": float(delivery_time_days),
        "category": category,
        "department": department,
        "brand": brand,
        "age": int(age),
        "traffic_source": traffic_source,
    }

    # Build DataFrame
    data = pd.DataFrame([item_dict])

    # Same encoding as training
    cat_cols = ["category", "department", "brand", "traffic_source"]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Align with training feature names
    data = data.reindex(columns=feature_names, fill_value=0)

    # Predict
    prob = model.predict_proba(data)[0][1]
    pct = prob * 100

    # Explain
    risk_label, reasons, suggestions = explain_prediction(item_dict, prob)

    st.subheader(f"Return risk: **{pct:.1f}%** ({risk_label})")

    st.progress(min(max(prob, 0.0), 1.0))

    if reasons:
        st.markdown("### 🧠 Why this risk level?")
        for r in reasons:
            st.markdown(f"- {r}")

    if suggestions:
        st.markdown("### 💡 Suggestions to reduce returns")
        for s in suggestions:
            st.markdown(f"- {s}")

    st.markdown("---")
    st.caption("Cloud-deployed version of the AI-Driven Return Prevention Platform capstone.")
else:
    st.info("Fill in the details above and click **Predict Return Risk**.")
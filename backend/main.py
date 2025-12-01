from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import xgboost as xgb
import pandas as pd
import os

app = FastAPI(
    title="AI Return Prevention API",
    description="Predict return risk + explanations + suggestions",
    version="1.0"
)

# ===============================
# 1. CORS (so frontend/Streamlit works)
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# 2. Load model + feature names
# ===============================
MODEL_PATH = "src/model/return_predictor.json"
FEATURE_PATH = "src/model/feature_names.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file missing → {MODEL_PATH}")

if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError(f"Feature list missing → {FEATURE_PATH}")

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

feature_names = joblib.load(FEATURE_PATH)

# ===============================
# 3. Pydantic schema
# ===============================
class CartItem(BaseModel):
    sale_price: float
    cost: float
    retail_price: float
    delivery_time_days: float
    category: str
    department: str
    brand: str
    age: int
    traffic_source: str

# ===============================
# 4. Explanation engine
# ===============================
def explain_prediction(item: CartItem, prob: float):
    """
    Simple rule-based explanation engine for return risk.
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

    # Rules based on your EDA
    if item.delivery_time_days > 10:
        reasons.append("Long delivery time may increase dissatisfaction.")
        suggestions.append("Offer faster shipping or set clearer expectations.")

    if item.retail_price > 0 and item.sale_price > item.retail_price:
        reasons.append("Sale price is higher than retail price — may confuse buyers.")
        suggestions.append("Review pricing to avoid inconsistencies.")

    if item.category.lower() in ["clothing", "shoes", "apparel"]:
        reasons.append("Category has high sizing/fit-related return rate.")
        suggestions.append("Add sizing charts, fit notes, and customer photos.")

    if item.age < 25:
        reasons.append("Younger customers tend to experiment more and return items.")
        suggestions.append("Highlight product details & return policy clearly.")

    if item.traffic_source.lower() in ["ads", "social", "affiliate"]:
        reasons.append("Ad-based visitors may impulse-buy and return more frequently.")
        suggestions.append("Improve landing page clarity & expectation setting.")

    if not reasons:
        reasons.append("No major risk factors detected.")
        suggestions.append("Maintain clear product descriptions and photos.")

    # Deduplicate suggestions
    suggestions = list(dict.fromkeys(suggestions))

    return {
        "risk_label": risk_label,
        "reasons": reasons,
        "suggestions": suggestions,
    }

# ===============================
# 5. Basic predict
# ===============================
@app.post("/predict")
def predict(item: CartItem):
    data = pd.DataFrame([item.dict()])

    cat_cols = ["category", "department", "brand", "traffic_source"]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Align with training columns
    data = data.reindex(columns=feature_names, fill_value=0)

    prob = model.predict_proba(data)[0][1]

    return {
        "return_probability": float(prob),
        "message": "Basic probability prediction (no explanations)."
    }

# ===============================
# 6. Predict + Explanation
# ===============================
@app.post("/predict_explained")
def predict_explained(item: CartItem):
    data = pd.DataFrame([item.dict()])

    # Encode categorical
    cat_cols = ["category", "department", "brand", "traffic_source"]
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Align with original training features
    data = data.reindex(columns=feature_names, fill_value=0)

    # Predict
    prob = model.predict_proba(data)[0][1]

    # Explain
    explanation = explain_prediction(item, prob)

    return {
        "return_probability": float(prob),
        "risk_label": explanation["risk_label"],
        "reasons": explanation["reasons"],
        "suggestions": explanation["suggestions"],
        "message": "Prediction + explanation using trained model."
    }

# ===============================
# 7. Health check
# ===============================
@app.get("/")
def root():
    return {"status": "API is running", "endpoints": ["/predict", "/predict_explained"]}
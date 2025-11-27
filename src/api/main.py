# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Return Prevention API", version="1.0")

# Load model & feature names once at startup
model = xgb.XGBClassifier()
model.load_model("../model/return_predictor.json")
feature_names = joblib.load("../model/feature_names.pkl")

class Item(BaseModel):
    sale_price: float
    cost: float
    retail_price: float
    delivery_time_days: float = 7
    category: str = "Tops"
    department: str = "Women"
    brand: str = "Allegra K"
    age: float = 35
    traffic_source: str = "Search"

@app.get("/")
def home():
    return {"message": "Return Prevention API is live! Go to /docs"}

@app.post("/predict")
def predict_return(item: Item):
    # Create DataFrame with correct columns
    input_data = pd.DataFrame([[
        item.sale_price,
        item.cost,
        item.retail_price,
        item.delivery_time_days,
        item.category,
        item.department,
        item.brand,
        item.age,
        item.traffic_source
    ]], columns=[
        'sale_price','cost','retail_price','delivery_time_days',
        'category','department','brand','age','traffic_source'
    ])

    # One-hot encode exactly like training time
    input_encoded = pd.get_dummies(input_data, columns=['category','department','brand','traffic_source'])
    input_encoded = input_encoded.reindex(columns=feature_names, fill_value=0)

    prob = model.predict_proba(input_encoded)[0][1]

    risk = "High Risk – Show AI Help" if prob > 0.5 else "Low Risk"

    return {
        "return_probability": round(float(prob), 3),
        "risk_level": risk,
        "recommendation": "Auto-enhance description + size guide" if prob > 0.5 else "Proceed normally"
    }

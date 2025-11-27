# app/demo.py
import streamlit as st
import requests
import json

st.set_page_config(page_title="Return Prevention Platform", layout="centered")
st.title("AI Return Prevention Platform")
st.caption("Capstone Project – Derek Karungani")

col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Sale Price ($)", 5, 300, 45)
    cost = st.number_input("Cost Price ($)", 0, 200, 20)
    days = st.slider("Delivery Days", 1, 30, 6)
with col2:
    category = st.selectbox("Category", ["Tops", "Bottoms", "Dresses", "Accessories", "Shoes"])
    department = st.selectbox("Department", ["Women", "Men"])
    brand = st.text_input("Brand", "Allegra K")

if st.button("Check Return Risk", type="primary"):
    payload = {
        "sale_price": price,
        "cost": cost,
        "retail_price": price + 15,
        "delivery_time_days": days,
        "category": category,
        "department": department,
        "brand": brand,
        "age": 32,
        "traffic_source": "Search"
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        result = response.json()

        if result["return_probability"] > 0.5:
            st.error(f"High Risk → {result['return_probability']:.0%} chance of return")
            st.warning(result["recommendation"])
            st.info("AI generating better size guide + 360° photos…")
        else:
            st.success(f"Low Risk → {result['return_probability']:.0%} chance")
            st.balloons()
    except:
        st.error("API not running → run:  uvicorn src.api.main:app --reload")

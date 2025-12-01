# AI-Driven Return Prevention Platform for E-Commerce

This project is my capstone for the MIT Emerging Talent Computer & Data Science Program.  
It extends our group work on predicting e-commerce returns and turns it into a **working prototype platform** that:

- Estimates the probability that an item will be **returned**  
- Classifies risk as **LOW / MEDIUM / HIGH**  
- Provides **simple explanations** of the risk  
- Suggests **concrete actions** to help prevent returns

---

## 1. Problem & Motivation

Online retailers often face return rates of 15–30%, which:

- Increases logistics, shipping, and restocking costs  
- Hurts customer satisfaction and loyalty  
- Wastes time and resources (and increases carbon footprint)

From our earlier group project, we saw that returns are often driven by:

- Vague or incomplete product descriptions  
- Unclear sizing and fit (especially in apparel and shoes)  
- Long or unexpected delivery times  
- Impulse purchases from ads and social media traffic

This capstone focuses on **turning those insights into a practical AI tool**.

---

## 2. What This Project Does

The system provides:

1. **ML-based return risk prediction**

   A trained **XGBoost classifier** takes in:

   - Item + price info  
   - Delivery time  
   - Product category/brand  
   - Customer age  
   - Traffic source  

   and outputs a **probability of return**.

2. **Risk label + explanations**

   Based on the probability and simple rules, it gives:

   - A risk label: **LOW / MEDIUM / HIGH**  
   - A list of human-readable **reasons**  
   - A list of **suggestions** for preventing returns

3. **Interactive interfaces**

   - A **FastAPI backend** with a `/predict_explained` endpoint  
   - A **simple HTML frontend** (`frontend/index.html`)  
   - A **Streamlit app** (`app.py`) for a more polished demo

---

## 3. Project Structure

```text
ai-return-prevention/
├── backend/
│   ├── __init__.py
│   └── main.py               # FastAPI app with /predict and /predict_explained
├── data/
│   ├── order_items.csv
│   ├── products.csv
│   └── users.csv
├── src/
│   └── model/
│       ├── return_predictor.json    # Trained XGBoost model
│       └── feature_names.pkl        # List of encoded feature names
├── notebooks/
│   └── 01_baseline_model.ipynb      # Training and feature engineering
├── frontend/
│   └── index.html            # Simple JavaScript + HTML UI
├── app.py                    # Streamlit app (dashboard-style UI)
├── requirements.txt
└── README.md

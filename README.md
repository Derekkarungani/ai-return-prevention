# AI-Driven E-Commerce Return Prevention Platform  
**MIT Emerging Talent – Capstone Project**  
**Author:** Derek Karungani

---

## 🔍 Overview

This project extends our earlier group work on predicting e-commerce product returns.  
Instead of stopping at analysis in a notebook, this capstone builds a **working AI system** that:

- Trains a machine learning model to predict whether an item will be returned  
- Serves predictions via a **FastAPI** backend  
- Provides a simple user interface using **Streamlit**  
- Demonstrates how AI can help reduce returns and improve customer satisfaction  

The main goal is to practice **end-to-end data science and ML deployment**:  
from raw data → model → API → user-facing demo.

---

## 🌐 Live Demo

👉 **Public Streamlit app:**  
`https://ai-return-prevention-mjuuxfjhputbzv9hzfdyqr.streamlit.app/

---

## 🧠 Key Features

- **Return Prediction Model**
  - XGBoost binary classifier trained on order, product, and user data  
  - Outputs a probability that an item will be returned  

- **Simple Explainability**
  - Uses feature values (price, delivery time, category, age, traffic source, etc.)  
  - Produces human-readable reasons and suggestions for each prediction  

- **API Backend (Local)**
  - FastAPI application (in `backend/main.py`)  
  - Endpoints for health check and prediction  
  - Swagger UI available at `http://127.0.0.1:8000/docs` when running locally  

- **Streamlit Frontend**
  - Local app (`app.py`) for interacting with the local API  
  - Cloud app (`app_cloud.py`) that talks directly to the model  
  - Simple form to enter product + customer info and view the predicted return risk  

---

## 🛠 Tech Stack

- **Language:** Python  
- **ML / Data:**  
  - pandas, numpy  
  - scikit-learn  
  - xgboost  
  - joblib  

- **Backend:**  
  - FastAPI  
  - Uvicorn  

- **Frontend / UI:**  
  - Streamlit  

- **Other:**  
  - Git & GitHub for version control  
  - Virtual environment (`.venv`) for dependency isolation  

---

## 📂 Project Structure

```text
ai-return-prevention/
│
├── backend/
│   └── main.py                # FastAPI application (local API)
│
├── frontend/                  # (placeholder for future UI work)
│
├── src/
│   └── model/
│       ├── return_predictor.json   # Trained XGBoost model
│       └── feature_names.pkl       # Feature list used by the model
│
├── data/                      # CSV files (orders, order_items, products, users)
│
├── notebooks/
│   └── 01_baseline_model.ipynb    # Model training and evaluation notebook
│
├── app.py                     # Local Streamlit app (talks to FastAPI)
├── app_cloud.py               # Cloud Streamlit app (loads model directly)
├── requirements.txt           # Python dependencies
└── README.md                  # This file

## Reflection & What I Learned

This project was much more than building a prediction model — it was an end-to-end experience in data engineering, modeling, software development, and deployment. As someone new to data science and still learning Python, this project pushed me far outside my comfort zone and forced me to understand how real systems are built.

I learned how to:
- structure a multi-folder production-style repository  
- clean, merge, and engineer features from large datasets  
- handle imbalanced targets and interpret performance metrics  
- build a machine-learning model using XGBoost  
- expose the model through a FastAPI backend  
- create a user-friendly Streamlit frontend  
- package everything together into a working prototype  
- manage virtual environments, requirements, and GitHub workflows  
- debug issues across Python, VS Code, terminal, and libraries  

One of my biggest challenges was working with memory-heavy datasets. I had to troubleshoot errors, sample data intelligently, reduce dimensionality, and iterate many times before I got a stable pipeline. Deploying the app locally and connecting all components taught me how real-world ML systems are designed.

Most importantly, this project helped me build confidence. In the beginning, I struggled with Python basics and felt intimidated by technical assignments. Completing this end-to-end system — model, API, and UI — showed me I can solve complex problems step by step. The experience strengthened my technical skills, my patience, and my ability to break challenges into manageable tasks.

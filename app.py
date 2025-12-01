import requests
import streamlit as st

# URL of your FastAPI backend
API_URL = "http://127.0.0.1:8000/predict_explained"

st.set_page_config(
    page_title="AI Return Risk Demo",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 AI-Driven E-Commerce Return Risk Demo")
st.write(
    "This app uses a trained XGBoost model (via FastAPI) to estimate the "
    "probability that an item will be returned and provides simple, "
    "rule-based explanations and suggestions to reduce that risk."
)

st.markdown("---")

# ===============================
# Input form
# ===============================
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
    payload = {
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

    with st.spinner("Contacting prediction API..."):
        try:
            res = requests.post(API_URL, json=payload, timeout=10)
        except Exception as e:
            st.error(f"❌ Request failed: {e}")
        else:
            if res.status_code != 200:
                st.error(f"❌ API error {res.status_code}: {res.text}")
            else:
                data = res.json()
                prob = data.get("return_probability", 0.0)
                pct = prob * 100
                risk = data.get("risk_label", "UNKNOWN")

                st.subheader(f"Return risk: **{pct:.1f}%** ({risk})")

                # Show a nice colored bar
                st.progress(min(max(prob, 0.0), 1.0))

                # Reasons
                reasons = data.get("reasons", [])
                suggestions = data.get("suggestions", [])

                if reasons:
                    st.markdown("### 🧠 Why the risk looks like this")
                    for r in reasons:
                        st.markdown(f"- {r}")

                if suggestions:
                    st.markdown("### 💡 Suggestions to prevent returns")
                    for s in suggestions:
                        st.markdown(f"- {s}")

                st.markdown("---")
                st.caption(
                    "Prototype built for the AI-Driven Return Prevention Platform capstone."
                )
else:
    st.info("Fill in the details above and click **Predict Return Risk**.")
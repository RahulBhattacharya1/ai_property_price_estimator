import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

st.set_page_config(page_title="Egypt Property Price Estimator", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "artifacts" / "model.joblib"
    return joblib.load(model_path)

model = load_model()

st.title("Egypt Real Estate Price Estimator")
st.caption("Predict price based on property details. Model trained on scraped listings.")

with st.form("predict_form"):
    col1, col2 = st.columns(2)

    with col1:
        prop_type = st.selectbox("Property Type", ["Apartment","Villa","Chalet","Townhouse","Duplex","Studio","Other"])
        size_sqm  = st.number_input("Size (sqm)", min_value=10.0, max_value=3000.0, value=150.0, step=10.0)
        bedrooms  = st.number_input("Bedrooms", min_value=0.0, max_value=20.0, value=3.0, step=1.0)

    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=20.0, value=2.0, step=1.0)
        payment   = st.selectbox("Payment Method", ["Cash","Installments","Mortgage","Other"])
        down_pay  = st.number_input("Down Payment (EGP)", min_value=0.0, max_value=1e9, value=0.0, step=10000.0)

    location_text    = st.text_input("Location (area, city)", "New Cairo, Cairo")
    description_text = st.text_area("Short Description", "Modern apartment with balcony and parking.")

    available_year  = st.number_input("Available Year", min_value=2000, max_value=2100, value=2025, step=1)
    available_month = st.number_input("Available Month", min_value=1, max_value=12, value=9, step=1)

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    # Build a single-row DataFrame matching training columns
    row = pd.DataFrame([{
        "type": prop_type,
        "payment_method": payment,
        "size_sqm": float(size_sqm),
        "bedrooms_num": float(bedrooms),
        "bathrooms_num": float(bathrooms),
        "down_payment_num": float(down_pay),
        "available_year": int(available_year),
        "available_month": int(available_month),
        "location": location_text,
        "description": description_text
    }])

    try:
        yhat = model.predict(row)[0]
        st.subheader(f"Estimated Price: {yhat:,.0f} EGP")
        st.caption("This estimate is based on historical listings and provided features.")

        # Simple sensitivity: +/-10% band (not a statistical CI, just a communication band)
        low, high = yhat*0.9, yhat*1.1
        st.write(f"Range: {low:,.0f} – {high:,.0f} EGP")

        st.divider()
        st.markdown("**Inputs used:**")
        st.json(row.to_dict(orient="records")[0])

    except Exception as e:
        st.error(f"Prediction failed: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title("🌾 AI-Based Crop Yield Prediction System")

st.write(
    "This system predicts crop yield and provides irrigation "
    "and fertilizer recommendations based on environmental conditions."
)

# -----------------------------
# User Inputs
# -----------------------------
year = st.number_input("Year", 1990, 2030, 2020)

rainfall = st.number_input(
    "Average Rainfall (mm/year)", min_value=0.0, max_value=5000.0, value=1200.0
)

pesticides = st.number_input(
    "Pesticides Used (tonnes)", min_value=0.0, max_value=50000.0, value=100.0
)

temp = st.number_input(
    "Average Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0
)

item = st.selectbox(
    "Crop Type",
    ["Maize", "Wheat", "Rice, paddy", "Potatoes", "Sorghum", "Sweet potatoes"]
)

area = st.selectbox(
    "Country",
    ["Bangladesh", "United Kingdom", "India", "USA"]
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Crop Yield"):

    # -----------------------------
    # Prepare input dataframe
    # -----------------------------
    input_dict = {
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temp
    }

    # Initialize all dummy columns
    for col in features:
        if col not in input_dict:
            input_dict[col] = 0

    # One-hot encode
    input_dict[f"Item_{item}"] = 1
    input_dict[f"Area_{area}"] = 1

    input_df = pd.DataFrame([input_dict])

    # Scale
    input_scaled = scaler.transform(input_df)

    # -----------------------------
    # Prediction
    # -----------------------------
    log_prediction = model.predict(input_scaled)[0]
    prediction = np.expm1(log_prediction)  # reverse log1p

    # -----------------------------
    # Irrigation Recommendation
    # -----------------------------
    if rainfall < 800:
        irrigation = "Low rainfall detected. Frequent irrigation is required."
    elif rainfall < 1200:
        irrigation = "Moderate rainfall. Maintain regular irrigation."
    else:
        irrigation = "Sufficient rainfall. Minimal irrigation required."

    if item == "Rice, paddy":
        irrigation += " Rice requires standing water during early growth stages."
    elif item in ["Wheat", "Maize"]:
        irrigation += " Irrigate during flowering and grain filling stages."

    # -----------------------------
    # Fertilizer Recommendation
    # -----------------------------
    if item == "Rice, paddy":
        fertilizer = "Use Urea and DAP. Apply Nitrogen in split doses."
    elif item == "Wheat":
        fertilizer = "Apply Nitrogen-rich NPK fertilizer during tillering."
    elif item == "Maize":
        fertilizer = "Use Nitrogen and Potassium fertilizers. Side dressing recommended."
    elif item == "Potatoes":
        fertilizer = "Apply Potassium-rich fertilizer to improve tuber quality."
    elif item == "Sorghum":
        fertilizer = "Use balanced NPK fertilizer."

    if temp > 30:
        fertilizer += " Avoid heavy fertilizer application during high temperatures."

    # -----------------------------
    # Output
    # -----------------------------
    st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} hg/ha")
    st.info(f"💧 Irrigation Recommendation: {irrigation}")
    st.info(f"🌿 Fertilizer Recommendation: {fertilizer}")

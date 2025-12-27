import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Initialize session state
# -----------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.irrigation = ""
    st.session_state.fertilizer = ""

# -----------------------------
# Load model artifacts
# -----------------------------
model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Crop Yield Prediction")
st.title("🌾 AI-Based Crop Yield Prediction System")

year = st.number_input("Year", 1990, 2030, 2020)
rainfall = st.number_input("Average Rainfall (mm/year)", 0.0, 5000.0, 1200.0)
pesticides = st.number_input("Pesticides Used (tonnes)", 0.0, 50000.0, 100.0)
temp = st.number_input("Average Temperature (°C)", 0.0, 50.0, 25.0)

item = st.selectbox(
    "Crop Type",
    ["Maize", "Wheat", "Rice, paddy", "Potatoes", "Sorghum", "Sweet potatoes"]
)

area = st.selectbox(
    "Country",
    ["Bangladesh", "United Kingdom", "India", "USA"]
)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Crop Yield"):

    input_dict = {
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temp
    }

    for col in features:
        if col not in input_dict:
            input_dict[col] = 0

    input_dict[f"Item_{item}"] = 1
    input_dict[f"Area_{area}"] = 1

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    log_pred = model.predict(input_scaled)[0]
    prediction = np.expm1(log_pred)

    # Irrigation
    if rainfall < 800:
        irrigation = "Low rainfall detected. Frequent irrigation required."
    elif rainfall < 1200:
        irrigation = "Moderate rainfall. Maintain regular irrigation."
    else:
        irrigation = "Sufficient rainfall. Minimal irrigation required."

    if item == "Rice, paddy":
        irrigation += " Rice requires standing water during early growth."
    elif item in ["Wheat", "Maize"]:
        irrigation += " Irrigate during flowering and grain filling."

    # Fertilizer
    if item == "Rice, paddy":
        fertilizer = "Use Urea and DAP. Apply Nitrogen in split doses."
    elif item == "Wheat":
        fertilizer = "Use Nitrogen-rich NPK during tillering."
    elif item == "Maize":
        fertilizer = "Apply Nitrogen and Potassium fertilizers."
    elif item == "Potatoes":
        fertilizer = "Use Potassium-rich fertilizer."
    else:
        fertilizer = "Use balanced NPK fertilizer."

    if temp > 30:
        fertilizer += " Avoid fertilizer application during high temperature."

    # Store results
    st.session_state.prediction = prediction
    st.session_state.irrigation = irrigation
    st.session_state.fertilizer = fertilizer

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
if st.session_state.prediction is not None:
    st.success(f"🌱 Predicted Crop Yield: {st.session_state.prediction:.2f} hg/ha")
    st.info(f"💧 Irrigation Recommendation: {st.session_state.irrigation}")
    st.info(f"🌿 Fertilizer Recommendation: {st.session_state.fertilizer}")

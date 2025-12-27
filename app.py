import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model and artifacts
# -----------------------------
model = joblib.load("lr_model.pkl")       # Your trained LR model
scaler = joblib.load("scaler.pkl")        # Scaler used during training
features = joblib.load("features.pkl")    # List of dummy columns

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title("🌾 AI-Based Crop Yield Prediction System")

st.write("""
This system predicts crop yield (hg/ha) and provides irrigation
and fertilizer recommendations based on crop type and environmental conditions.
""")

# -----------------------------
# User Inputs
# -----------------------------
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

    # Prepare input dictionary
    input_dict = {
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temp
    }

    # Initialize dummy columns to 0
    for col in features:
        if col not in input_dict:
            input_dict[col] = 0

    # Set one-hot encoding for selected crop and area
    input_dict[f"Item_{item}"] = 1
    input_dict[f"Area_{area}"] = 1

    input_df = pd.DataFrame([input_dict])

    # Scale numeric features
    input_scaled = scaler.transform(input_df)

    # Predict (reverse log1p)
    log_pred = model.predict(input_scaled)[0]
    prediction = np.expm1(log_pred)

    # -----------------------------
    # Irrigation Recommendation
    # -----------------------------
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

    # -----------------------------
    # Fertilizer Recommendation
    # -----------------------------
    if item == "Rice, paddy":
        fertilizer = "Use Urea and DAP. Apply Nitrogen in split doses."
    elif item == "Wheat":
        fertilizer = "Use Nitrogen-rich NPK during tillering."
    elif item == "Maize":
        fertilizer = "Apply Nitrogen and Potassium fertilizers."
    elif item == "Potatoes":
        fertilizer = "Use Potassium-rich fertilizer."
    elif item == "Sorghum":
        fertilizer = "Use balanced NPK fertilizer."
    elif item == "Sweet potatoes":
        fertilizer = "Apply balanced NPK fertilizer."

    # Adjust for high temperature
    if temp > 30:
        fertilizer += " Avoid fertilizer application during high temperature."

    # -----------------------------
    # Display Results
    # -----------------------------
    st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} hg/ha")
    st.info(f"💧 Irrigation Recommendation: {irrigation}")
    st.info(f"🌿 Fertilizer Recommendation: {fertilizer}")

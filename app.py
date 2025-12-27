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

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")
st.title("🌾 AI-Based Crop Yield Prediction System")

st.write("""
This system predicts crop yield (hg/ha) and provides irrigation,
fertilizer, and management recommendations based on environmental conditions.
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
# Function to generate detailed recommendations
# -----------------------------
def generate_recommendations(yield_value, rainfall, temperature, pesticides):
    recommendations = []

    # Yield-based
    if yield_value < 20000:
        recommendations.append("⚠️ Low yield predicted. Consider soil testing and better seed varieties.")
        recommendations.append("🌱 Improve irrigation and fertilizer management.")
    elif yield_value < 50000:
        recommendations.append("✅ Moderate yield predicted. Optimize water and nutrient usage.")
    else:
        recommendations.append("🎉 High yield predicted. Maintain current best practices.")

    # Rainfall-based
    if rainfall < 800:
        recommendations.append("💧 Low rainfall detected. Use drip irrigation or water conservation.")
    elif rainfall > 2000:
        recommendations.append("🌧️ High rainfall detected. Ensure proper drainage.")

    # Temperature-based
    if temperature > 35:
        recommendations.append("🌡️ High temperature may affect crops. Consider heat-resistant varieties.")
    elif temperature < 15:
        recommendations.append("❄️ Low temperature detected. Crop growth may slow.")

    # Pesticide usage
    if pesticides > 100:
        recommendations.append("🧪 High pesticide usage detected. Reduce to avoid soil damage.")
    else:
        recommendations.append("🧪 Pesticide usage is within safe limits.")

    return recommendations

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Crop Yield"):

    # Prepare input
    input_dict = {
        "Year": year,
        "average_rain_fall_mm_per_year": rainfall,
        "pesticides_tonnes": pesticides,
        "avg_temp": temp
    }

    # Initialize dummy columns
    for col in features:
        if col not in input_dict:
            input_dict[col] = 0

    # One-hot encode
    input_dict[f"Item_{item}"] = 1
    input_dict[f"Area_{area}"] = 1

    input_df = pd.DataFrame([input_dict])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict and reverse log1p
    log_pred = model.predict(input_scaled)[0]
    prediction = np.expm1(log_pred)

    # -----------------------------
    # Irrigation
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
    # Fertilizer
    # -----------------------------
    if item == "Rice, paddy":
        fertilizer = "Use Urea and DAP. Apply Nitrogen in split doses."
    elif item == "Wheat":
        fertilizer = "Use Nitrogen-rich NPK during tillering."
    elif item == "Maize":
        fertilizer = "Apply Nitrogen and Potassium fertilizers."
    elif item == "Potatoes":
        fertilizer = "Use Potassium-rich fertilizer."
    elif item in ["Sorghum", "Sweet potatoes"]:
        fertilizer = "Use balanced NPK fertilizer."

    if temp > 30:
        fertilizer += " Avoid fertilizer application during high temperature."

    # -----------------------------
    # Generate additional recommendations
    # -----------------------------
    additional_recs = generate_recommendations(prediction, rainfall, temp, pesticides)

    # -----------------------------
    # Display Results
    # -----------------------------
    st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} hg/ha")
    st.info(f"💧 Irrigation Recommendation: {irrigation}")
    st.info(f"🌿 Fertilizer Recommendation: {fertilizer}")

    st.subheader("📌 Additional Recommendations:")
    for rec in additional_recs:
        st.write(f"- {rec}")

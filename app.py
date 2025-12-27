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
# Generate recommendations function
# -----------------------------
def generate_recommendations(yield_value, rainfall, temperature, pesticides):
    recommendations = []

    if yield_value < 20000:
        recommendations.append("⚠️ Low yield predicted. Consider soil testing and better seed varieties.")
        recommendations.append("🌱 Improve irrigation and fertilizer management.")
    elif yield_value < 50000:
        recommendations.append("✅ Moderate yield predicted. Optimize water and nutrient usage.")
    else:
        recommendations.append("🎉 High yield predicted. Maintain current best practices.")

    if rainfall < 800:
        recommendations.append("💧 Low rainfall detected. Use drip irrigation or water conservation.")
    elif rainfall > 2000:
        recommendations.append("🌧️ High rainfall detected. Ensure proper drainage.")

    if temperature > 35:
        recommendations.append("🌡️ High temperature may affect crops. Consider heat-resistant varieties.")
    elif temperature < 15:
        recommendations.append("❄️ Low temperature detected. Crop growth may slow.")

    if pesticides > 100:
        recommendations.append("🧪 High pesticide usage detected. Reduce to avoid soil damage.")
    else:
        recommendations.append("🧪 Pesticide usage is within safe limits.")

    return recommendations

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Crop Yield"):

    # Prepare input dataframe
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
    input_scaled = scaler.transform(input_df)

    # Prediction
    log_pred = model.predict(input_scaled)[0]
    prediction = np.expm1(log_pred)  # reverse log1p

    # -----------------------------
    # Generate recommendations only using your function
    recs = generate_recommendations(prediction, rainfall, temp, pesticides)

    # -----------------------------
    # Display results
    st.success(f"🌱 Predicted Crop Yield: {prediction:.2f} hg/ha")
    st.subheader("📌 Recommendations:")
    for r in recs:
        st.write(f"- {r}")

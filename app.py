import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model, scaler, features
# -----------------------------
model = joblib.load("lr_model.pkl")     # CHANGED
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AI-Based Crop Yield Prediction")

st.write("Enter the details to predict crop yield (hg/ha):")

# Numeric inputs
year = st.number_input("Year", min_value=1990, max_value=2030, value=2020)
rainfall = st.number_input("Average Rainfall (mm/year)", value=1200.0)
pesticides = st.number_input("Pesticides (tonnes)", value=100.0)
temp = st.number_input("Average Temperature (°C)", value=25.0)

# Dropdowns
item = st.selectbox(
    "Select Crop",
    ["Maize", "Wheat", "Rice, paddy", "Potatoes", "Sorghum", "Sweet potatoes"]
)

area = st.selectbox(
    "Select Country",
    ["Bangladesh", "United Kingdom", "India", "USA"]
)

# -----------------------------
# Prepare input
# -----------------------------
input_dict = {
    "Year": year,
    "average_rain_fall_mm_per_year": rainfall,
    "pesticides_tonnes": pesticides,
    "avg_temp": temp
}

# Initialize all feature columns as 0
for col in features:
    if col not in input_dict:
        input_dict[col] = 0

# One-hot encoding
item_col = f"Item_{item}"
area_col = f"Area_{area}"

if item_col in features:
    input_dict[item_col] = 1

if area_col in features:
    input_dict[area_col] = 1

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Scaling (VERY IMPORTANT)
# -----------------------------
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_scaled)[0]

# ❗ Use this ONLY if you trained with log1p(y)
prediction = np.expm1(prediction)

# -----------------------------
# Output
# -----------------------------
st.success(f"Predicted Crop Yield: {prediction:.2f} hg/ha")

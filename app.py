import streamlit as st
import numpy as np

# Import your prediction function
# (you will implement this in src/inference/predict.py)
from src.inference.predict import predict

# -------------------------------
# UI TITLE
# -------------------------------
st.set_page_config(page_title="Battery BMS AI", layout="centered")

st.title("🔋 Battery Management System (AI)")
st.write("Predict battery health / anomalies using AI model")

# -------------------------------
# USER INPUT SECTION
# -------------------------------
st.header("Input Parameters")

voltage = st.number_input("Voltage (V)", min_value=0.0, value=3.7)
current = st.number_input("Current (A)", value=1.0)
temperature = st.number_input("Temperature (°C)", value=25.0)

# Example additional inputs (modify based on your dataset)
soc = st.slider("State of Charge (SOC %)", 0, 100, 50)

# Convert input to array
input_data = np.array([voltage, current, temperature, soc])

# -------------------------------
# PREDICTION BUTTON
# -------------------------------
if st.button("Predict"):
    try:
        result = predict(input_data)

        st.success("Prediction complete ✅")

        st.subheader("Result:")
        st.write(result)

    except Exception as e:
        st.error(f"Error: {e}")

# -------------------------------
# OPTIONAL VISUALIZATION
# -------------------------------
st.subheader("Input Summary")
st.write({
    "Voltage": voltage,
    "Current": current,
    "Temperature": temperature,
    "SOC": soc
})
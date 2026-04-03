import streamlit as st
import subprocess

st.set_page_config(page_title="BMS AI System", layout="centered")

st.title("🔋 Battery Management System")

# -----------------------
# INPUTS
# -----------------------
soc = st.slider("SoC", 0.0, 1.0, 0.45)
soh = st.slider("SoH", 0.0, 1.0, 0.95)
temp = st.number_input("Temperature (°C)", value=27.0)
current = st.number_input("Current (A)", value=-1.5)

mode = st.selectbox(
    "Mode",
    ["auto", "fast", "balanced", "battery_care"]
)

# -----------------------
# RUN PIPELINE
# -----------------------
if st.button("Run Pipeline"):

    with st.spinner("Running full BMS pipeline..."):

        cmd = [
            "python", "predict.py",
            "--soc", str(soc),
            "--soh", str(soh),
            "--temp", str(temp),
            "--current", str(current),
            "--mode", mode
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

    st.success("Done ✅")

    # -----------------------
    # OUTPUT
    # -----------------------
    st.subheader("Pipeline Output")

    if result.stdout:
        st.text(result.stdout)

    if result.stderr:
        st.error(result.stderr)

import streamlit as st
import torch
import pickle
import os
from src.bms_pipeline import BatteryTransformer, run_predictor, run_simulator_optimiser, run_meta_agent, run_kill_agent
#import (
 #   BatteryTransformer,
  #  run_predictor,
   # run_simulator_optimiser,
    #run_meta_agent,
     #run_kill_agent
#)

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
# PATHS (relative)
# -----------------------
MODEL_PATH = "models/best_model.pt"
GLOBALS_PATH = "models/predictor_globals.pkl"

# -----------------------
# RUN PIPELINE
# -----------------------
if st.button("Run Pipeline"):

    with st.spinner("Running AI pipeline..."):

        device = torch.device("cpu")

        # Load model
        model = BatteryTransformer(input_dim=11).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # Load globals
        with open(GLOBALS_PATH, "rb") as f:
            globs = pickle.load(f)

        global_mean = globs["global_mean"]
        global_std = globs["global_std"]

        battery_input = {
            "soc": soc,
            "soh": soh,
            "temp_C": temp,
            "current_A": current,
            "cycle_norm": 0.5
        }

        # ---- AGENT 1
        predictor_output = run_predictor(
            battery_input, model, global_mean, global_std, device
        )

        # ---- AGENT 2
        df, transformer_state = run_simulator_optimiser(predictor_output)

        transformer_state["confidence"] = predictor_output["confidence"]

        # ---- AGENT 3
        selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
            df, transformer_state, mode=mode
        )

        # ---- AGENT 4
        final_policy, decision = run_kill_agent(
            df, selected_policy, transformer_state, policies, metrics_df
        )

    st.success("Done ✅")

    # -----------------------
    # OUTPUT
    # -----------------------
    st.subheader("🔮 Predictor Output")
    st.write(predictor_output)

    st.subheader("🧠 Decision")
    st.write(decision)

    st.subheader("⚡ Final Policy")
    st.write(final_policy)

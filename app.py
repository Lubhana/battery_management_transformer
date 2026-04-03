import streamlit as st
import torch
import pickle
import os
import time
import numpy as np
import pandas as pd

# Fixed import path: importing directly from bms_pipeline
from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
    extract_policy,
    compute_metrics
)

# -----------------------
# PAGE CONFIG & STYLING
# -----------------------
st.set_page_config(page_title="BMS AI Simulation", page_icon="🔋", layout="wide")
st.title("🔋 Continuous Battery Management System Simulator")
st.markdown("Visualizing AI-driven charging policies, ECM states, and RUL estimation.")

# -----------------------
# SIDEBAR INPUTS
# -----------------------
st.sidebar.header("Initial Battery State")
soc = st.sidebar.slider("SoC", 0.0, 1.0, 0.45)
soh = st.sidebar.slider("SoH", 0.0, 1.0, 0.95)
temp = st.sidebar.number_input("Temperature (°C)", value=27.0)
current = st.sidebar.number_input("Initial Current (A)", value=-1.5)

st.sidebar.header("Agent Settings")
mode = st.sidebar.selectbox(
    "Meta-Agent Mode",
    ["auto", "fast", "balanced", "battery_care"]
)

# Paths (relative)
MODEL_PATH = "models/best_model.pt"
GLOBALS_PATH = "models/predictor_globals.pkl"

# -----------------------
# MAIN EXECUTION LOGIC
# -----------------------
if st.sidebar.button("Run Simulation", type="primary"):
    
    # Check if models exist before running
    if not os.path.exists(MODEL_PATH) or not os.path.exists(GLOBALS_PATH):
        st.error(f"Missing model files. Please ensure {MODEL_PATH} and {GLOBALS_PATH} exist.")
        st.stop()

    with st.spinner("Initializing Multi-Agent Pipeline..."):
        device = torch.device("cpu")

        # Load model and globals
        model = BatteryTransformer(input_dim=11).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
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

        # 1. PREDICTOR
        predictor_output = run_predictor(battery_input, model, global_mean, global_std, device)
        # FORCE the predictor to obey the UI sliders:



        # 2. SIMULATOR
        df, transformer_state = run_simulator_optimiser(predictor_output)
        transformer_state["confidence"] = predictor_output["confidence"]

        # >>> PATCH 1: SAFETY CHECK PREVENTS KEYERROR CRASH <<<
        if df.empty or "solution_id" not in df.columns:
            st.error("🚨 Critical Safety Abort: The battery state is too extreme. The simulator could not find any safe charging profiles to generate a dataset.")
            st.stop() 
        # >>> ---------------------------------------------- <<<

        # 3. META-AGENT
        selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
            df, transformer_state, mode=mode
        )

        # 4. KILL AGENT
        final_policy, decision = run_kill_agent(
            df, selected_policy, transformer_state, policies, metrics_df
        )

    st.success("Pipeline Execution Complete.")

    # -----------------------
    # RESULTS DASHBOARD
    # -----------------------
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted SoC", f"{predictor_output['soc']:.2%}")
    col2.metric("Predicted SoH", f"{predictor_output['soh']:.2%}")
    col3.metric("Predicted Temp", f"{predictor_output['temperature']:.1f} °C")
    col4.metric("Model Confidence", f"{predictor_output['confidence']:.2%}")

    st.subheader("🛡️ Kill Agent Status")
    status_color = "green" if decision["decision"] == "allow" else "orange" if decision["decision"] == "override" else "red"
    st.markdown(f"**Decision:** :{status_color}[{decision['decision'].upper()}] — {decision['reason']}")

    # -----------------------
    # VISUALIZATION & RUL
    # -----------------------
    if final_policy is not None:
        policy_data = extract_policy(df, final_policy)
        metrics = compute_metrics(policy_data)
        
        # Calculate Remaining Useful Life (RUL) 
        soh_loss_per_cycle = metrics["soh_loss"]
        current_soh = predictor_output['soh']
        end_of_life_threshold = 0.80
        
        # >>> PATCH 2: FIXES THE RUL == 0 ISSUE <<<
        st.subheader("🔋 Lifecycle & RUL Analysis")
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Policy ID", int(final_policy))
        c2.metric("Cycle SoH Degradation", f"{soh_loss_per_cycle:.6f}")
        
        if soh_loss_per_cycle > 0 and current_soh > end_of_life_threshold:
            projected_rul_cycles = int((current_soh - end_of_life_threshold) / soh_loss_per_cycle)
            c3.metric("Estimated RUL (Cycles)", projected_rul_cycles, help="Cycles remaining until SoH hits 80%")
        else:
            c3.metric("Estimated RUL", "Infinite (No Damage)")
        # >>> --------------------------------- <<<

        st.divider()
        st.subheader("📈 Live Charging Simulation")
        
        # Setup empty charts for continuous streaming
        chart_col1, chart_col2, chart_col3 = st.columns(3)
        
        with chart_col1:
            st.markdown("**State of Charge (SoC)**")
            soc_chart = st.line_chart([policy_data['soc'][0]], height=250)
            
        with chart_col2:
            st.markdown("**Temperature (K)**")
            temp_chart = st.line_chart([policy_data['temp'][0]], height=250, color="#ffaa00")
            
        with chart_col3:
            st.markdown("**Applied Current (A)**")
            curr_chart = st.line_chart([policy_data['current'][0]], height=250, color="#ff0000")

        # Simulate continuous real-time data output
        st.caption("Simulating real-time BMS execution...")
        for i in range(1, len(policy_data['soc'])):
            soc_chart.add_rows([policy_data['soc'][i]])
            temp_chart.add_rows([policy_data['temp'][i]])
            curr_chart.add_rows([policy_data['current'][i]])
            time.sleep(0.02) 

    else:
        st.error("Charging Aborted by Kill Agent. No policy to simulate.")

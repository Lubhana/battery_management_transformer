import streamlit as st
import torch
import pickle
import os
import time
import numpy as np
import pandas as pd

from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
    extract_policy,
    compute_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMS AI Simulation",
    page_icon="🔋",
    layout="wide",
)

st.title("🔋 Continuous Battery Management System Simulator")
st.markdown(
    "Visualizing AI-driven charging policies, ECM states, and RUL estimation."
)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR INPUTS
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Initial Battery State")

soc = st.sidebar.slider("SoC", 0.0, 1.0, 0.45, step=0.01,
                         help="State of Charge (0 = empty, 1 = full)")
soh = st.sidebar.slider("SoH", 0.0, 1.0, 0.95, step=0.01,
                         help="State of Health (1 = new, 0 = dead)")
temp    = st.sidebar.number_input("Temperature (°C)", value=27.0, step=0.5)
current = st.sidebar.number_input(
    "Initial Current (A)", value=-1.5, step=0.1,
    help="Negative = discharging, Positive = charging"
)
cycle_norm = st.sidebar.slider(
    "Cycle Index (normalised)", 0.0, 1.0, 0.5, step=0.01,
    help="0 = brand-new battery, 1 = heavily cycled"
)

st.sidebar.header("Agent Settings")
mode = st.sidebar.selectbox(
    "Meta-Agent Mode",
    ["auto", "fast", "balanced", "battery_care"],
    help="auto = AI decides based on SoC/SoH; others force a strategy",
)

MODEL_PATH   = "models/best_model.pt"
GLOBALS_PATH = "models/predictor_globals.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_globals(model_path, globals_path, device):
    """Cache model loading so it doesn't reload on every rerun."""
    model = BatteryTransformer(input_dim=11).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with open(globals_path, "rb") as f:
        globs = pickle.load(f)
    return model, globs["global_mean"], globs["global_std"]


def make_line_df(values, col_name):
    """
    Wrap a 1-D array into a single-column DataFrame.
    st.line_chart / add_rows needs a DataFrame or dict, not a bare list.
    """
    return pd.DataFrame({col_name: [values] if np.isscalar(values) else values})


def stream_charts(policy_data, batch=10):
    """
    Stream SoC / Temp / Current charts in batches to avoid blocking the
    Streamlit event loop with thousands of individual add_rows calls.
    """
    n = len(policy_data["soc"])

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    with chart_col1:
        st.markdown("**State of Charge (SoC)**")
        soc_chart = st.line_chart(
            make_line_df(policy_data["soc"][0], "SoC"), height=250
        )
    with chart_col2:
        st.markdown("**Temperature (K)**")
        temp_chart = st.line_chart(
            make_line_df(policy_data["temp"][0], "Temp (K)"),
            height=250, color="#ffaa00",
        )
    with chart_col3:
        st.markdown("**Applied Current (A)**")
        curr_chart = st.line_chart(
            make_line_df(policy_data["current"][0], "Current (A)"),
            height=250, color="#ff4444",
        )

    st.caption("Simulating real-time BMS execution…")
    progress = st.progress(0)

    for i in range(1, n, batch):
        end = min(i + batch, n)
        soc_chart.add_rows(make_line_df(policy_data["soc"][i:end],   "SoC"))
        temp_chart.add_rows(make_line_df(policy_data["temp"][i:end],  "Temp (K)"))
        curr_chart.add_rows(make_line_df(policy_data["current"][i:end], "Current (A)"))
        progress.progress(end / n)
        time.sleep(0.05)

    progress.empty()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("Run Simulation", type="primary"):

    # ── pre-flight checks ────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH) or not os.path.exists(GLOBALS_PATH):
        st.error(
            f"Missing model files. "
            f"Please ensure **{MODEL_PATH}** and **{GLOBALS_PATH}** exist."
        )
        st.stop()

    device = torch.device("cpu")

    # ── load (cached inside the button block so it only runs when needed) ────
    with st.spinner("Loading model weights…"):
        try:
            model, global_mean, global_std = load_model_and_globals(
                MODEL_PATH, GLOBALS_PATH, device
            )
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    battery_input = {
        "soc":        soc,
        "soh":        soh,
        "temp_C":     temp,
        "current_A":  current,
        "cycle_norm": cycle_norm,   # FIX: was hardcoded 0.5
    }

    # ── Agent 1: Predictor ───────────────────────────────────────────────────
    with st.spinner("Agent 1 — Predictor running…"):
        try:
            predictor_output = run_predictor(
                battery_input, model, global_mean, global_std, device
            )
        except Exception as e:
            st.error(f"Predictor failed: {e}")
            st.stop()

    # ── Agent 2: Simulator + Optimiser ───────────────────────────────────────
    with st.spinner("Agent 2 — Simulator & Optimiser running…"):
        try:
            df, transformer_state = run_simulator_optimiser(predictor_output)
            transformer_state["confidence"] = predictor_output["confidence"]
        except Exception as e:
            st.error(f"Simulator failed: {e}")
            st.stop()

    if df is None or df.empty or "solution_id" not in df.columns:
        st.error(
            "🚨 Critical Safety Abort: The battery state is too extreme. "
            "The simulator could not generate any safe charging profiles."
        )
        st.stop()

    # ── Agent 3: Meta-Agent ──────────────────────────────────────────────────
    with st.spinner("Agent 3 — Meta-Agent selecting policy…"):
        try:
            selected_policy, policies, metrics_df, policy_choices = run_meta_agent(
                df, transformer_state, mode=mode
            )
        except Exception as e:
            st.error(f"Meta-Agent failed: {e}")
            st.stop()

    # ── Agent 4: Kill Agent ──────────────────────────────────────────────────
    with st.spinner("Agent 4 — Kill Agent performing safety checks…"):
        try:
            final_policy, decision = run_kill_agent(
                df, selected_policy, transformer_state, policies, metrics_df
            )
        except Exception as e:
            st.error(f"Kill Agent failed: {e}")
            st.stop()

    st.success("Pipeline Execution Complete.")

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()

    # ── top metrics ──────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # FIX: show both the user's input value AND the AI's prediction so the
    # user can see how far the model deviated from the slider input.
    col1.metric(
        "Predicted SoC",
        f"{predictor_output['soc']:.2%}",
        delta=f"{predictor_output['soc'] - soc:+.2%} vs input",
        delta_color="off",
    )
    col2.metric(
        "Predicted SoH",
        f"{predictor_output['soh']:.2%}",
        delta=f"{predictor_output['soh'] - soh:+.2%} vs input",
        delta_color="off",
    )
    col3.metric(
        "Predicted Temp",
        f"{predictor_output['temperature']:.1f} °C",
        delta=f"{predictor_output['temperature'] - temp:+.1f} °C vs input",
        delta_color="off",
    )
    col4.metric(
        "Model Confidence",
        f"{predictor_output['confidence']:.2%}",
    )

    # ── confidence warning ───────────────────────────────────────────────────
    if predictor_output["confidence"] < 0.5:
        st.warning(
            "⚠️ Model confidence is low (<50%). "
            "The battery state may be outside the training distribution. "
            "Predictions should be treated with caution."
        )

    # ── per-target confidences ────────────────────────────────────────────────
    with st.expander("Per-target confidence breakdown"):
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("SoC confidence",  f"{predictor_output['soc_conf']:.2%}")
        cc2.metric("SoH confidence",  f"{predictor_output['soh_conf']:.2%}")
        cc3.metric("Temp confidence", f"{predictor_output['temp_conf']:.2%}")

    # ── kill agent status ────────────────────────────────────────────────────
    st.subheader("🛡️ Kill Agent Status")

    decision_text = decision["decision"].upper()
    if decision["decision"] == "allow":
        st.success(f"✅ **{decision_text}** — {decision['reason']}")
    elif decision["decision"] == "override":
        st.warning(f"⚠️ **{decision_text}** — {decision['reason']}")
    else:
        st.error(f"🚨 **{decision_text}** — {decision['reason']}")

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE, RUL & CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    if final_policy is not None:
        policy_data  = extract_policy(df, final_policy)
        metrics      = compute_metrics(policy_data)

        soh_loss_per_cycle   = metrics["soh_loss"]
        current_soh          = predictor_output["soh"]
        end_of_life_threshold = 0.80

        st.subheader("🔋 Lifecycle & RUL Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Policy ID",     int(final_policy))
        c2.metric("SoC Gain",               f"{metrics['soc_gain']:.4f}")
        c3.metric("Cycle SoH Degradation",  f"{soh_loss_per_cycle:.6f}")
        c4.metric("Peak Temperature",       f"{metrics['peak_temp']:.1f} K")

        # RUL
        st.divider()
        rul_col1, rul_col2 = st.columns([1, 2])
        with rul_col1:
            if soh_loss_per_cycle > 1e-9 and current_soh > end_of_life_threshold:
                projected_rul = int(
                    (current_soh - end_of_life_threshold) / soh_loss_per_cycle
                )
                st.metric(
                    "Estimated RUL (Cycles)",
                    f"{projected_rul:,}",
                    help="Cycles remaining until SoH hits 80% (end-of-life threshold)",
                )
            else:
                st.metric("Estimated RUL", "∞ — No measurable damage")

        with rul_col2:
            # Mini SoH degradation projection chart
            if soh_loss_per_cycle > 1e-9:
                max_cycles = min(projected_rul + 50, 5000)
                cycle_range = np.arange(0, max_cycles, max(1, max_cycles // 200))
                soh_proj    = current_soh - soh_loss_per_cycle * cycle_range
                soh_proj    = np.clip(soh_proj, 0, 1)
                proj_df     = pd.DataFrame({"Projected SoH": soh_proj}, index=cycle_range)
                st.markdown("**SoH Degradation Projection**")
                st.line_chart(proj_df, height=150, color="#00cc88")

        # ── live streaming charts ────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Live Charging Simulation")
        stream_charts(policy_data, batch=20)

    else:
        st.error(
            "🚨 Charging Aborted by Kill Agent. "
            "No safe policy could be found for the current battery state."
        )

        # Still show the predictor output summary even on abort
        st.info(
            f"**Predictor summary:** "
            f"SoC={predictor_output['soc']:.2%}, "
            f"SoH={predictor_output['soh']:.2%}, "
            f"Temp={predictor_output['temperature']:.1f}°C, "
            f"Confidence={predictor_output['confidence']:.2%}"
        )

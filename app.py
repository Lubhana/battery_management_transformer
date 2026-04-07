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
    model = BatteryTransformer(input_dim=11).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with open(globals_path, "rb") as f:
        globs = pickle.load(f)
    return model, globs["global_mean"], globs["global_std"]


def make_line_df(values, col_name):
    return pd.DataFrame({col_name: [values] if np.isscalar(values) else values})


def stream_charts(policy_data, batch=10):
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
        soc_chart.add_rows(make_line_df(policy_data["soc"][i:end],     "SoC"))
        temp_chart.add_rows(make_line_df(policy_data["temp"][i:end],   "Temp (K)"))
        curr_chart.add_rows(make_line_df(policy_data["current"][i:end],"Current (A)"))
        progress.progress(end / n)
        time.sleep(0.05)

    progress.empty()


def layman_battery_grade(soh):
    if soh >= 0.95:
        return "🟢 Excellent", "Your battery is almost brand new."
    elif soh >= 0.85:
        return "🟢 Good", "Your battery is in great shape with plenty of life left."
    elif soh >= 0.75:
        return "🟡 Fair", "Your battery is aging but still functional."
    elif soh >= 0.60:
        return "🟠 Poor", "Your battery has significant wear — consider monitoring closely."
    else:
        return "🔴 Critical", "Your battery is heavily degraded and may need replacement soon."


def layman_charge_level(soc):
    if soc >= 0.80:
        return "nearly full"
    elif soc >= 0.50:
        return "half full"
    elif soc >= 0.25:
        return "getting low"
    else:
        return "almost empty"


def layman_temp(temp_c):
    if temp_c < 15:
        return "cold (which slightly reduces performance)"
    elif temp_c <= 35:
        return "normal"
    elif temp_c <= 45:
        return "warm (safe but worth watching)"
    else:
        return "hot (this can accelerate battery wear)"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────
if st.sidebar.button("Run Simulation", type="primary"):

    if not os.path.exists(MODEL_PATH) or not os.path.exists(GLOBALS_PATH):
        st.error(
            f"Missing model files. "
            f"Please ensure **{MODEL_PATH}** and **{GLOBALS_PATH}** exist."
        )
        st.stop()

    device = torch.device("cpu")

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
        "cycle_norm": cycle_norm,
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
            df, transformer_state, nsga_info = run_simulator_optimiser(
                predictor_output, battery_input
            )
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

    if predictor_output["confidence"] < 0.5:
        st.warning(
            "⚠️ Model confidence is low (<50%). "
            "The battery state may be outside the training distribution. "
            "Predictions should be treated with caution."
        )

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
    # NSGA-II OPTIMISER EXPLAINER
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧬 Optimisation Engine — What Happened Under the Hood")

    with st.expander("How the AI found the best charging strategy — click to expand", expanded=True):
        ni = nsga_info

        if ni["source"] == "cached":
            st.info(
                f"**Pre-computed profiles loaded from cache.** "
                f"The simulator found {ni['n_solutions']} ready-to-use charging strategies "
                f"from a previous NSGA-II run and loaded them instantly."
            )
        else:
            st.success(
                f"**Fresh optimisation completed.** "
                f"The AI ran {ni['algorithm']} and discovered "
                f"**{ni['n_solutions']} optimal charging strategies** for your battery."
            )

        st.markdown("#### What is NSGA-II?")
        st.markdown(
            """
**NSGA-II** (Non-dominated Sorting Genetic Algorithm II) is an evolutionary optimiser —
it works a bit like natural selection, but for charging strategies.

Here's how it works in plain English:

1. **Start with random strategies** — the AI generates 60 random charging profiles (different
   patterns of how much current to apply over 20 minutes).

2. **Simulate each one** — every strategy is run through a physics model of your battery
   (the ECM) to see what would actually happen: how much charge you'd gain, how hot it gets,
   and how much wear it causes.

3. **Survival of the fittest** — strategies that are unsafe, too hot, or too damaging get
   discarded. The better ones survive and are "bred" together to create new strategies.

4. **Repeat for 40 generations** — after 40 rounds of this, the AI has a set of
   strategies that represent the best possible trade-offs.

5. **The Pareto front** — rather than picking just one "winner", NSGA-II keeps a whole
   *family* of optimal strategies — some charge faster, some are gentler on the battery,
   and some balance both. This family is called the **Pareto front**.
            """
        )

        st.markdown("#### The 3 goals it balanced simultaneously")
        g1, g2, g3 = st.columns(3)
        g1.info("⚡ **Maximise SoC Gain**\nCharge as much as possible in the time window.")
        g2.info("🌡️ **Minimise Peak Temperature**\nKeep the battery cool to avoid thermal stress.")
        g3.info("🔋 **Minimise SoH Loss**\nCause as little wear as possible per charge cycle.")

        if ni["source"] == "fresh" and ni["soc_gain_range"]:
            st.markdown("#### What the Pareto front looked like")
            r1, r2, r3 = st.columns(3)
            r1.metric("SoC gain range",
                      f"{ni['soc_gain_range'][0]:.3f} → {ni['soc_gain_range'][1]:.3f}")
            r2.metric("Peak temp range",
                      f"{ni['peak_temp_range'][0]:.1f} K → {ni['peak_temp_range'][1]:.1f} K")
            r3.metric("SoH loss range",
                      f"{ni['soh_loss_range'][0]:.6f} → {ni['soh_loss_range'][1]:.6f}")
            st.caption(
                "Each number shows the spread across all Pareto-optimal strategies. "
                "The Meta-Agent then picked the one best suited to your battery's current state."
            )

        st.markdown(
            f"**Run config:** {ni['algorithm']} · "
            f"{ni['pop_size']} individuals · "
            f"{ni['generations']} generations · "
            f"{ni['objectives']} objectives · "
            f"{ni['n_solutions']} Pareto-optimal solutions"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE, RUL & CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    if final_policy is not None:
        policy_data  = extract_policy(df, final_policy)
        metrics      = compute_metrics(policy_data)

        soh_loss_per_cycle    = metrics["soh_loss"]
        current_soh           = predictor_output["soh"]
        end_of_life_threshold = 0.80

        st.subheader("🔋 Lifecycle & RUL Analysis")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Selected Policy ID",    int(final_policy))
        c2.metric("SoC Gain",              f"{metrics['soc_gain']:.4f}")
        c3.metric("Cycle SoH Degradation", f"{soh_loss_per_cycle:.6f}")
        c4.metric("Peak Temperature",      f"{metrics['peak_temp']:.1f} K")

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
                projected_rul = None
                st.metric("Estimated RUL", "∞ — No measurable damage")

        with rul_col2:
            if soh_loss_per_cycle > 1e-9:
                max_cycles  = min(projected_rul + 50, 5000) if projected_rul else 5000
                cycle_range = np.arange(0, max_cycles, max(1, max_cycles // 200))
                soh_proj    = np.clip(current_soh - soh_loss_per_cycle * cycle_range, 0, 1)
                proj_df     = pd.DataFrame({"Projected SoH": soh_proj}, index=cycle_range)
                st.markdown("**SoH Degradation Projection**")
                st.line_chart(proj_df, height=150, color="#00cc88")

        # ── live streaming charts ────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Live Charging Simulation")
        stream_charts(policy_data, batch=20)

        # ─────────────────────────────────────────────────────────────────────
        # LAYMAN SUMMARY
        # ─────────────────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📋 Plain-English Summary")
        st.markdown("*Everything the simulation just did, explained simply.*")

        battery_grade, grade_desc = layman_battery_grade(predictor_output["soh"])
        charge_level  = layman_charge_level(predictor_output["soc"])
        temp_desc     = layman_temp(predictor_output["temperature"])
        conf_pct      = predictor_output["confidence"]
        decision_word = decision["decision"]

        # ── Battery health card ──
        with st.container(border=True):
            st.markdown(f"### {battery_grade} — Battery Health")
            st.markdown(
                f"{grade_desc} The AI estimates your battery's health (SoH) at "
                f"**{predictor_output['soh']:.1%}**, meaning it can hold "
                f"**{predictor_output['soh']:.1%} of its original capacity** compared "
                f"to when it was new."
            )
            if predictor_output.get("ood"):
                st.warning(
                    "⚠️ The AI flagged your battery inputs as unusual — "
                    "a combination it wasn't trained on. It played it safe and used "
                    "your raw input values directly instead of making a potentially "
                    "unreliable prediction."
                )

        # ── Current state card ──
        with st.container(border=True):
            st.markdown("### 🔌 Current Battery State")
            st.markdown(
                f"Right now your battery is **{charge_level}** "
                f"({predictor_output['soc']:.1%} charged) and running at a "
                f"**{temp_desc}** temperature of {predictor_output['temperature']:.1f} °C."
            )
            if conf_pct >= 0.75:
                conf_text = "The AI is **highly confident** in these readings."
            elif conf_pct >= 0.5:
                conf_text = "The AI has **moderate confidence** in these readings — treat them as a good estimate."
            else:
                conf_text = ("⚠️ The AI has **low confidence** in these readings. "
                             "Your battery may be in an unusual state the model hasn't seen before.")
            st.markdown(conf_text)

        # ── Charging decision card ──
        with st.container(border=True):
            st.markdown("### ⚡ Charging Decision")
            if decision_word == "allow":
                st.success(
                    f"**Charging is safe to proceed.** The AI reviewed the best strategy "
                    f"(Policy #{int(final_policy)}) and found no safety concerns. "
                    f"It will charge your battery by approximately "
                    f"**{metrics['soc_gain']:.1%}** during this session."
                )
            elif decision_word == "override":
                st.warning(
                    f"**The original strategy was adjusted.** The AI's safety agent "
                    f"detected a potential issue ({decision['reason']}) and automatically "
                    f"switched to a safer, gentler charging profile "
                    f"(Policy #{int(final_policy)})."
                )
            else:
                st.error(
                    "**Charging was stopped.** The safety agent determined that no "
                    "available charging strategy was safe enough for your battery's "
                    f"current condition ({decision['reason']}). "
                    "It's best not to charge right now."
                )

        # ── Wear & longevity card ──
        with st.container(border=True):
            st.markdown("### 🕰️ Battery Wear This Session")
            wear_pct = soh_loss_per_cycle * 100
            st.markdown(
                f"This charging session will use up roughly **{wear_pct:.4f}%** "
                f"of your battery's total lifetime health — an extremely small amount."
            )
            if projected_rul is not None:
                st.markdown(
                    f"At this rate, your battery has approximately "
                    f"**{projected_rul:,} charge cycles remaining** before it drops "
                    f"below 80% of its original capacity, which is the standard "
                    f"end-of-life point for most lithium-ion batteries."
                )
                if projected_rul > 1000:
                    longevity = "That's a very healthy lifespan — your battery should last a long time."
                elif projected_rul > 300:
                    longevity = "That's a reasonable lifespan for a battery at this health level."
                else:
                    longevity = "The battery is nearing end-of-life and may need replacement relatively soon."
                st.markdown(longevity)
            else:
                st.markdown(
                    "The damage this session is so small it's essentially **unmeasurable**. "
                    "Your battery is in excellent shape."
                )

        # ── Temperature card ──
        with st.container(border=True):
            st.markdown("### 🌡️ Temperature During Charging")
            peak_k = metrics["peak_temp"]
            peak_c = peak_k - 273.15
            if peak_c <= 35:
                temp_verdict = "✅ The battery stayed **cool throughout** — ideal charging conditions."
            elif peak_c <= 45:
                temp_verdict = "🟡 The battery got **moderately warm** — within safe limits but worth monitoring."
            else:
                temp_verdict = "🔴 The battery got **quite hot** during charging — this can accelerate wear over time."
            st.markdown(
                f"The peak temperature reached during this charging session was "
                f"**{peak_c:.1f} °C** ({peak_k:.1f} K). {temp_verdict}"
            )

    else:
        # ── Abort path ───────────────────────────────────────────────────────
        st.error(
            "🚨 Charging Aborted by Kill Agent. "
            "No safe policy could be found for the current battery state."
        )
        st.info(
            f"**Predictor summary:** "
            f"SoC={predictor_output['soc']:.2%}, "
            f"SoH={predictor_output['soh']:.2%}, "
            f"Temp={predictor_output['temperature']:.1f}°C, "
            f"Confidence={predictor_output['confidence']:.2%}"
        )

        # Layman abort summary
        st.divider()
        st.subheader("📋 Plain-English Summary")
        battery_grade, grade_desc = layman_battery_grade(predictor_output["soh"])
        with st.container(border=True):
            st.markdown(f"### {battery_grade} — Why Charging Was Stopped")
            st.markdown(
                f"{grade_desc} However, the AI's safety system determined that "
                f"**none of the available charging strategies were safe enough** "
                f"for your battery in its current state.\n\n"
                f"This can happen when the battery is very degraded, extremely hot, "
                f"or in an unusual condition the AI hasn't encountered before. "
                f"The system chose to do nothing rather than risk damaging your battery further."
            )
            st.markdown(
                "**What you can do:** Let the battery cool down, check that your "
                "input values are correct, or consult a battery health diagnostic tool."
            )

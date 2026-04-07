import streamlit as st
import torch
import pickle
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.bms_pipeline import (
    BatteryTransformer,
    run_predictor,
    run_simulator_optimiser,
    run_meta_agent,
    run_kill_agent,
    extract_policy,
    compute_metrics,
    build_input_sequence,
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
# XAI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def xai_predictor(battery_input, model, global_mean, global_std, device,
                  predictor_output):
    """
    Sensitivity-based feature importance for Agent 1 (Predictor).

    For each of the 5 user-facing input features, we perturb it by ±10%,
    re-run the transformer, and measure how much SoC and SoH predictions move.
    The average absolute shift is the feature's importance score.
    """
    features = {
        "SoC (charge level)":     ("soc",        0.05),
        "SoH (battery health)":   ("soh",        0.05),
        "Temperature (°C)":       ("temp_C",     2.0),
        "Current (A)":            ("current_A",  0.2),
        "Cycle index":            ("cycle_norm", 0.05),
    }

    base_soc = predictor_output["soc"]
    base_soh = predictor_output["soh"]

    importances_soc = {}
    importances_soh = {}

    model.eval()
    for label, (key, delta) in features.items():
        shifts_soc, shifts_soh = [], []
        for sign in [+1, -1]:
            perturbed = dict(battery_input)
            perturbed[key] = battery_input[key] + sign * delta
            # clip to valid ranges
            if key in ("soc", "soh", "cycle_norm"):
                perturbed[key] = float(np.clip(perturbed[key], 0.01, 0.99))

            seq = build_input_sequence(perturbed, global_mean, global_std)
            x   = torch.tensor(seq).unsqueeze(0).to(device)
            with torch.no_grad():
                soc_out, soh_out, _ = model(x)
            shifts_soc.append(abs(float(soc_out[0, 0]) - base_soc))
            shifts_soh.append(abs(float(soh_out[0, 0]) - base_soh))

        importances_soc[label] = float(np.mean(shifts_soc))
        importances_soh[label] = float(np.mean(shifts_soh))

    return importances_soc, importances_soh


def xai_meta_agent(transformer_state, mode, policy_choices, selected_policy):
    """
    Rule-weight explanation for Agent 3 (Meta-Agent).

    Returns a dict of {factor: score} showing how much each factor
    contributed to — or constrained — the final strategy selection.
    Score is normalised so they sum to 1.
    """
    soc  = transformer_state["soc"]
    soh  = transformer_state["soh"]
    conf = transformer_state.get("confidence", 1.0)

    factors = {}

    # Confidence gate — overrides everything if low
    factors["Model confidence\n(low → forces gentle)"] = max(0.0, 1.0 - conf)

    # Manual mode override
    factors["Manual mode override\n(you chose the mode)"] = 0.0 if mode == "auto" else 1.0

    # SoH influence — how far below 0.9 it is
    factors["Battery health (SoH)\n(low → prefers gentle)"] = max(0.0, 0.9 - soh) / 0.9

    # SoC urgency — how empty the battery is
    factors["Charge level (SoC)\n(low → prefers fast)"] = max(0.0, 0.4 - soc) / 0.4

    # SoH conflict with SoC — degraded battery despite low charge
    if soc < 0.4 and soh < 0.9:
        factors["Degraded + low SoC\n(conflict → balanced)"] = 0.5
    else:
        factors["Degraded + low SoC\n(conflict → balanced)"] = 0.0

    total = sum(factors.values()) or 1.0
    return {k: v / total for k, v in factors.items()}


def xai_kill_agent(metrics, battery_state, decision,
                   soh_loss_limit, health_limit,
                   peak_temp_limit=320, temp_rise_limit=5,
                   high_temp_limit=5, confidence_limit=0.5):
    """
    Rule-proximity explanation for Agent 4 (Kill Agent).

    Returns a list of dicts: each rule, its current value, its limit,
    how close it was to breaching (0 = fine, 1 = exactly at limit, >1 = breached),
    and whether it was breached.
    """
    conf = battery_state.get("confidence", 1.0)

    rules = [
        {
            "rule":     "Peak Temperature",
            "value":    metrics["peak_temp"],
            "limit":    peak_temp_limit,
            "unit":     "K",
            "outcome":  "abort",
            "plain":    "battery got too hot",
        },
        {
            "rule":     "Rapid Temp Rise",
            "value":    metrics["temp_rise"],
            "limit":    temp_rise_limit,
            "unit":     "K/step",
            "outcome":  "abort",
            "plain":    "temperature spiked too fast",
        },
        {
            "rule":     "Sustained Overheat",
            "value":    metrics["high_temp_duration"],
            "limit":    high_temp_limit,
            "unit":     "steps >315 K",
            "outcome":  "override",
            "plain":    "battery stayed hot for too long",
        },
        {
            "rule":     "SoH Loss",
            "value":    metrics["soh_loss"],
            "limit":    soh_loss_limit,
            "unit":     "ΔSoH",
            "outcome":  "override",
            "plain":    "too much battery wear",
        },
        {
            "rule":     "Battery Health",
            "value":    1 - battery_state["soh"],
            "limit":    1 - health_limit,
            "unit":     "degradation",
            "outcome":  "override",
            "plain":    "battery already too degraded",
        },
        {
            "rule":     "Predictor Confidence",
            "value":    1 - conf,
            "limit":    1 - confidence_limit,
            "unit":     "uncertainty",
            "outcome":  "override",
            "plain":    "AI wasn't confident enough",
        },
    ]

    for r in rules:
        r["ratio"]    = r["value"] / (r["limit"] + 1e-12)
        r["breached"] = r["value"] > r["limit"]

    return rules


def plot_importance_chart(importance_dict, title, color):
    """Render a horizontal bar chart of importances using matplotlib."""
    labels = list(importance_dict.keys())
    values = list(importance_dict.values())

    # Sort descending
    pairs  = sorted(zip(values, labels), reverse=True)
    values = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(labels) * 0.55)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(labels, values, color=color, alpha=0.85)
    ax.set_xlabel("Importance (sensitivity)", color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=10, pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top", "right", "bottom", "left"]].set_color("#444")
    ax.xaxis.label.set_color("white")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", color="white", fontsize=8)

    plt.tight_layout()
    return fig


def plot_kill_agent_chart(rules):
    """Render the kill agent rule proximity chart."""
    labels  = [r["rule"] for r in rules]
    ratios  = [min(r["ratio"], 1.5) for r in rules]   # cap at 1.5× for display
    colors  = ["#ff4444" if r["breached"] else
               "#ffaa00" if r["ratio"] > 0.75 else
               "#00cc88"
               for r in rules]

    fig, ax = plt.subplots(figsize=(6, max(2.5, len(rules) * 0.55)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.barh(labels, ratios, color=colors, alpha=0.85)
    ax.axvline(x=1.0, color="white", linewidth=1.2, linestyle="--", alpha=0.7,
               label="Safety limit")
    ax.set_xlabel("Value as fraction of limit  (1.0 = at limit)", color="white", fontsize=9)
    ax.set_title("Kill Agent — Rule Proximity", color="white", fontsize=10, pad=8)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[["top", "right", "bottom", "left"]].set_color("#444")

    legend_patches = [
        mpatches.Patch(color="#00cc88", label="Safe"),
        mpatches.Patch(color="#ffaa00", label="Near limit (>75%)"),
        mpatches.Patch(color="#ff4444", label="Breached"),
    ]
    ax.legend(handles=legend_patches, loc="lower right",
              facecolor="#1a1a2e", labelcolor="white", fontsize=7)

    plt.tight_layout()
    return fig


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
    # META-AGENT DECISION — which policy was picked and why
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🤖 Meta-Agent Decision")

    # Figure out which named bucket (fast/balanced/gentle) was selected
    policy_name = None
    for name, pid in policy_choices.items():
        if pid == selected_policy:
            policy_name = name
            break
    policy_name = policy_name or "custom"

    # Pull that policy's metrics for display
    sel_row = metrics_df[metrics_df["solution_id"] == selected_policy].iloc[0]

    policy_icons = {"fast": "⚡", "balanced": "⚖️", "gentle": "🌿"}
    policy_descs = {
        "fast":     "Charges as quickly as possible. Best when SoC is very low and the battery is healthy.",
        "balanced": "A middle-ground strategy — reasonable charge speed with moderate wear.",
        "gentle":   "Slow and careful charging. Prioritises battery longevity over speed.",
    }
    icon = policy_icons.get(policy_name, "🔧")
    desc = policy_descs.get(policy_name, "A custom strategy selected from the Pareto front.")

    # Retrieve the reason string from meta_agent_select via metrics_df
    # (re-derive it cleanly from transformer_state so we don't need to change the pipeline)
    soc_val  = transformer_state["soc"]
    soh_val  = transformer_state["soh"]
    conf_val = transformer_state.get("confidence", 1.0)

    if conf_val < 0.5:
        derived_reason = f"Model confidence is low ({conf_val:.0%}) — the AI defaulted to the gentlest strategy to stay safe."
    elif mode != "auto":
        derived_reason = f"You manually selected **{mode}** mode, so the AI followed your instruction."
    elif soh_val < 0.9:
        derived_reason = f"Battery health (SoH) is {soh_val:.0%}, which is below 90% — the AI chose gentle to protect the degraded battery."
    elif soc_val < 0.4 and soh_val >= 0.9:
        derived_reason = f"Battery is low ({soc_val:.0%} charge) but healthy ({soh_val:.0%} SoH) — the AI chose fast to top it up quickly."
    elif soc_val < 0.4 and soh_val < 0.9:
        derived_reason = f"Battery is low ({soc_val:.0%} charge) AND somewhat degraded ({soh_val:.0%} SoH) — the AI balanced speed with care."
    else:
        derived_reason = "Battery is in a normal state with no special conditions — the AI defaulted to the balanced strategy."

    dec_col1, dec_col2 = st.columns([1, 2])
    with dec_col1:
        st.markdown(f"### {icon} {policy_name.capitalize()} Charging")
        st.markdown(desc)
        st.markdown(f"**Policy ID selected:** `{int(selected_policy)}`")

    with dec_col2:
        st.info(f"**Why this strategy?**\n\n{derived_reason}")
        m1, m2, m3 = st.columns(3)
        m1.metric("SoC Gain",     f"{sel_row['soc_gain']:.4f}",
                  help="How much charge this policy adds")
        m2.metric("Peak Temp",    f"{sel_row['peak_temp']:.1f} K",
                  help="Hottest the battery gets under this policy")
        m3.metric("SoH Loss",     f"{sel_row['soh_loss']:.6f}",
                  help="Battery wear caused by this charging session")

    # Show all three candidate policies for comparison
    with st.expander("Compare all three candidate strategies from the Pareto front"):
        rows = []
        for name, pid in policy_choices.items():
            row = metrics_df[metrics_df["solution_id"] == pid].iloc[0]
            rows.append({
                "Strategy":   f"{policy_icons[name]} {name.capitalize()}",
                "Policy ID":  int(pid),
                "SoC Gain":   f"{row['soc_gain']:.4f}",
                "Peak Temp (K)": f"{row['peak_temp']:.1f}",
                "SoH Loss":   f"{row['soh_loss']:.6f}",
                "Selected":   "✅" if pid == selected_policy else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
    # XAI — EXPLAINABILITY DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Explainability (XAI) — Why Did the AI Decide This?")
    st.markdown(
        "Each agent in the pipeline made a decision. "
        "Here's exactly what drove those decisions, with charts and plain-English explanations."
    )

    xai_tab1, xai_tab2, xai_tab3 = st.tabs([
        "🧠 Agent 1 — Predictor",
        "🤖 Agent 3 — Meta-Agent",
        "🛡️ Agent 4 — Kill Agent",
    ])

    # ── XAI Tab 1: Predictor ─────────────────────────────────────────────────
    with xai_tab1:
        st.markdown("### Why did the AI predict this SoC and SoH?")
        st.markdown(
            "We perturbed each of your input values slightly (±small amount) and "
            "measured how much the prediction changed. A bigger change means that "
            "input had more influence on the result."
        )

        if predictor_output.get("ood"):
            st.warning(
                "⚠️ OOD bypass was active — the transformer was not used, "
                "so sensitivity analysis is not applicable. "
                "The prediction came directly from your raw inputs."
            )
        else:
            with st.spinner("Running sensitivity analysis…"):
                imp_soc, imp_soh = xai_predictor(
                    battery_input, model, global_mean, global_std,
                    device, predictor_output
                )

            c1, c2 = st.columns(2)
            with c1:
                fig_soc = plot_importance_chart(
                    imp_soc, "Feature Importance for SoC Prediction", "#4e9af1"
                )
                st.pyplot(fig_soc, use_container_width=True)
                plt.close(fig_soc)

            with c2:
                fig_soh = plot_importance_chart(
                    imp_soh, "Feature Importance for SoH Prediction", "#00cc88"
                )
                st.pyplot(fig_soh, use_container_width=True)
                plt.close(fig_soh)

            # Plain-English interpretation
            top_soc = max(imp_soc, key=imp_soc.get)
            top_soh = max(imp_soh, key=imp_soh.get)
            with st.container(border=True):
                st.markdown("#### What this means in plain English")
                st.markdown(
                    f"- The **SoC prediction** ({predictor_output['soc']:.1%}) was most "
                    f"sensitive to **{top_soc}** — meaning if that value changed slightly, "
                    f"the AI's charge estimate would shift the most.\n"
                    f"- The **SoH prediction** ({predictor_output['soh']:.1%}) was most "
                    f"sensitive to **{top_soh}** — that input had the strongest influence "
                    f"on the battery health estimate.\n"
                    f"- Features with near-zero bars had almost no effect on the prediction — "
                    f"the AI effectively ignored them for this input."
                )

    # ── XAI Tab 2: Meta-Agent ────────────────────────────────────────────────
    with xai_tab2:
        st.markdown("### Why did the AI pick this charging strategy?")
        st.markdown(
            "The Meta-Agent weighs several factors when choosing between Fast, "
            "Balanced, and Gentle charging. The chart below shows which factors "
            "drove the decision and how strongly."
        )

        factor_weights = xai_meta_agent(
            transformer_state, mode, policy_choices, selected_policy
        )

        # Compute adaptive thresholds so the kill-agent explanation is consistent
        input_soh = transformer_state["soh"]
        if input_soh < 0.5:
            ka_soh_loss_limit = 0.0003
            ka_health_limit   = 0.50
        elif input_soh < 0.65:
            ka_soh_loss_limit = 0.0005
            ka_health_limit   = 0.55
        else:
            ka_soh_loss_limit = 0.001
            ka_health_limit   = 0.80

        c1, c2 = st.columns([1, 1])
        with c1:
            fig_meta = plot_importance_chart(
                factor_weights, "Meta-Agent Decision Factors", "#ffaa00"
            )
            st.pyplot(fig_meta, use_container_width=True)
            plt.close(fig_meta)

        with c2:
            # Show the actual values that fed into each factor
            st.markdown("#### Input values that drove this decision")
            factor_table = pd.DataFrame([
                {"Factor": "SoC (charge level)",     "Value": f"{transformer_state['soc']:.1%}",  "Threshold": "< 40% → prefers fast"},
                {"Factor": "SoH (battery health)",   "Value": f"{transformer_state['soh']:.1%}",  "Threshold": "< 90% → prefers gentle"},
                {"Factor": "Model confidence",        "Value": f"{transformer_state.get('confidence',1):.1%}", "Threshold": "< 50% → forces gentle"},
                {"Factor": "Charging mode",           "Value": mode,                                "Threshold": "non-auto → follows your choice"},
            ])
            st.dataframe(factor_table, use_container_width=True, hide_index=True)

        top_factor = max(factor_weights, key=factor_weights.get)
        with st.container(border=True):
            st.markdown("#### What this means in plain English")
            policy_icons = {"fast": "⚡", "balanced": "⚖️", "gentle": "🌿"}
            icon = policy_icons.get(policy_name, "🔧")
            st.markdown(
                f"The AI chose **{icon} {policy_name.capitalize()}** charging. "
                f"The single biggest reason was **{top_factor.split(chr(10))[0]}**, "
                f"which accounted for {factor_weights[top_factor]:.0%} of the decision weight.\n\n"
                f"{derived_reason}"
            )
            if all(v < 0.05 for v in factor_weights.values()):
                st.info(
                    "All factors had near-zero weight — the battery is in a "
                    "normal, healthy state, so the AI defaulted to Balanced charging."
                )

    # ── XAI Tab 3: Kill Agent ────────────────────────────────────────────────
    with xai_tab3:
        st.markdown("### Why did the safety agent allow / block charging?")
        st.markdown(
            "The Kill Agent checks 6 safety rules before allowing charging to proceed. "
            "Each bar below shows how close the selected policy came to breaching that rule. "
            "**Reaching 1.0 means the limit was hit.** Anything above 1.0 is a breach."
        )

        if final_policy is not None:
            ka_policy  = extract_policy(df, final_policy)
            ka_metrics = compute_metrics(ka_policy)
            ka_battery_state = {
                "soc":        transformer_state["soc"],
                "soh":        transformer_state["soh"],
                "temp":       transformer_state["temp"],
                "confidence": transformer_state.get("confidence", 1.0),
            }

            rules = xai_kill_agent(
                ka_metrics, ka_battery_state, decision,
                soh_loss_limit=ka_soh_loss_limit,
                health_limit=ka_health_limit,
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                fig_kill = plot_kill_agent_chart(rules)
                st.pyplot(fig_kill, use_container_width=True)
                plt.close(fig_kill)

            with c2:
                st.markdown("#### Rule-by-rule breakdown")
                for r in rules:
                    status = "🔴 BREACHED" if r["breached"] else \
                             "🟡 Near limit" if r["ratio"] > 0.75 else "🟢 Safe"
                    st.markdown(
                        f"**{r['rule']}** — {status}  \n"
                        f"Value: `{r['value']:.5f}` / Limit: `{r['limit']:.5f}` "
                        f"({r['ratio']:.1%} of limit)"
                    )

            with st.container(border=True):
                st.markdown("#### What this means in plain English")
                breached = [r for r in rules if r["breached"]]
                near     = [r for r in rules if not r["breached"] and r["ratio"] > 0.75]
                safe     = [r for r in rules if not r["breached"] and r["ratio"] <= 0.75]

                if decision["decision"] == "allow":
                    st.success(
                        f"✅ **All {len(rules)} safety rules passed.** "
                        f"The selected charging strategy is well within safe limits. "
                        + (f"{len(near)} rule(s) were moderately close to their limits "
                           f"({', '.join(r['rule'] for r in near)}) but none were breached."
                           if near else "All rules had comfortable margins.")
                    )
                elif decision["decision"] == "override":
                    st.warning(
                        f"⚠️ **{len(breached)} rule(s) were breached** on the originally "
                        f"selected policy: {', '.join(r['rule'] for r in breached)}. "
                        f"The Kill Agent automatically switched to a safer policy. "
                        f"This policy ({int(final_policy)}) passes all rules."
                    )
                else:
                    st.error(
                        f"🚨 **{len(breached)} critical rule(s) were breached** and no safe "
                        f"alternative could be found: "
                        f"{', '.join(r['plain'] for r in breached)}. Charging was stopped."
                    )

                # Closest rule to the limit
                closest = max(rules, key=lambda r: r["ratio"])
                st.markdown(
                    f"**Closest call:** *{closest['rule']}* reached "
                    f"**{closest['ratio']:.1%}** of its safety limit "
                    f"({closest['value']:.5f} vs limit {closest['limit']:.5f})."
                )
        else:
            st.error("No final policy available — charging was aborted before rule evaluation.")

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

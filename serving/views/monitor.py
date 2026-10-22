"""Model and monitoring: every evaluation output, in one place."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from console import figure, read_table, reports_dir


def section(title: str) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def missing(command: str) -> None:
    st.markdown(f'<p class="note">Not produced yet. Run <b>{command}</b>.</p>',
                unsafe_allow_html=True)


def show_table(name: str, command: str, fmt: dict | None = None) -> pd.DataFrame | None:
    table = read_table(name)
    if table is None:
        missing(command)
        return None
    st.dataframe(table.style.format(fmt or {}, na_rep="—"), hide_index=True,
                 use_container_width=True)
    return table


def show_figures(names: list[str], width: int = 2) -> None:
    present = [(n, figure(n)) for n in names if figure(n)]
    if not present:
        return
    cols = st.columns(width)
    for k, (_, path) in enumerate(present):
        cols[k % width].image(str(path), use_container_width=True)


st.markdown('<div class="runline">Evaluation &nbsp;&middot;&nbsp; held-out test span '
            '&nbsp;&middot;&nbsp; chronological protocol</div>', unsafe_allow_html=True)
st.markdown('<h1 class="masthead">Model and monitoring</h1>', unsafe_allow_html=True)

summary_path = reports_dir() / "metrics" / "evaluation_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

section("Detection · model comparison (AUC-PR is the primary metric)")
show_table("model_comparison.csv", "python -m frds evaluate", {
    c: "{:.4f}" for c in ("auc_pr", "auc_roc", "f1_best", "mcc", "brier",
                          "recall_at_fpr_0.001", "recall_at_fpr_0.01", "recall_at_top_0.005")
} | {"auc_pr_lift_over_prevalence": "{:.1f}"})
show_figures(["pr_curves.png", "model_comparison.png"])

section("Calibration and risk tiers")
c1, c2 = st.columns(2, gap="large")
with c1:
    show_figures(["reliability.png"], width=1)
with c2:
    st.markdown('<p class="note">Tiers as specified (0–29 / 30–69 / 70–100)</p>',
                unsafe_allow_html=True)
    show_table("risk_tiers_config.csv", "python -m frds evaluate")
    st.markdown('<p class="note">Tiers derived from the cost model and review capacity</p>',
                unsafe_allow_html=True)
    show_table("risk_tiers_derived.csv", "python -m frds evaluate")
cost = summary.get("cost", {}).get("config")
if cost:
    saved = 1 - cost["total_cost"] / max(cost["cost_if_nothing_flagged"], 1e-9)
    st.markdown(
        f'<p class="note">Modelled loss under the specified tiers: '
        f'<b>{cost["total_cost"]:,.0f}</b> against <b>{cost["cost_if_nothing_flagged"]:,.0f}</b> '
        f"if nothing were flagged - a reduction of <b>{saved:.1%}</b>.</p>",
        unsafe_allow_html=True)

section("Explanation · what the model relies on, and whether the reasons are faithful")
e1, e2 = st.columns(2, gap="large")
with e1:
    show_figures(["shap_groups.png", "shap_beeswarm.png"], width=1)
with e2:
    show_table("shap_group_importance.csv", "python -m frds evaluate",
               {"mean_abs_shap": "{:.4f}", "share": "{:.1%}"})
    st.markdown('<p class="note">Deletion test: remove an alert\'s top reasons vs the same '
                "number of random features, and measure how far the score falls. Faithful "
                "reasons move the score far more.</p>", unsafe_allow_html=True)
    show_table("explanation_faithfulness.csv", "python -m frds evaluate")

section("Drift · population stability of the score across the test span")
d1, d2 = st.columns(2, gap="large")
with d1:
    show_figures(["psi_drift.png"], width=1)
with d2:
    show_table("psi_score_drift.csv", "python -m frds evaluate")

section("Serving cost · latency")
st.markdown('<p class="note">Product path: one raw transaction through online features, '
            "the deployed model, calibration and the decision; reasons timed separately.</p>",
            unsafe_allow_html=True)
show_table("latency_product.csv", "python -m frds evaluate")
st.markdown('<p class="note">Reference: every enabled tier plus the stack, scoring '
            "precomputed feature rows.</p>", unsafe_allow_html=True)
show_table("latency.csv", "python -m frds evaluate")

section("Robustness · evasion attack on the deployed model")
show_table("robustness.csv", "python -m frds robustness")
show_figures(["robustness.png"], width=2)

section("Transfer · the same protocol on Bank Account Fraud (BAF)")
show_table("baf_transfer.csv", "python -m frds baf")

section("Tuning · Optuna search on the validation span")
tuned = sorted((reports_dir() / "tables").glob("tuning_*.csv")) if (reports_dir() / "tables").exists() else []
if not tuned:
    missing("python -m frds tune")
for path in tuned:
    st.markdown(f'<p class="note">{path.stem.replace("tuning_", "")}</p>', unsafe_allow_html=True)
    st.dataframe(pd.read_csv(path).head(10), hide_index=True, use_container_width=True)

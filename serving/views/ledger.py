"""Live ledger: the decision layer, watched while it runs."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from console import TIER_VERB, get_scorer, read_table, reference_scores
from frds.config import load_config
from frds.evaluation.drift import interpret, psi
from replay import QueueState, ReplayStream
from theme import band_rows, figure_line, ledger_style, plain_reasons

DRIFT_WINDOW = 2_000


def init_state(scorer, batch_size: int) -> None:
    if "stream" not in st.session_state:
        st.session_state.stream = ReplayStream(scorer.test_positions(), batch_size)
        st.session_state.queue = QueueState()
        st.session_state.playing = False
        st.session_state.selected = None
    st.session_state.stream.batch_size = batch_size


def advance(scorer) -> None:
    stream: ReplayStream = st.session_state.stream
    batch = stream.next_batch()
    if not len(batch):
        st.session_state.playing = False
        return
    started = time.perf_counter()
    decisions = scorer.score_positions(batch)
    st.session_state.queue.extend(decisions, (time.perf_counter() - started) * 1000.0)


scorer = get_scorer()
info = scorer.describe()
cfg = load_config()

with st.sidebar:
    st.markdown('<div class="runline">Replay</div>', unsafe_allow_html=True)
    batch_size = st.select_slider("Batch size", options=[64, 128, 256, 512, 1024], value=256)
    init_state(scorer, batch_size)

    a, b = st.columns(2)
    if a.button("Play", use_container_width=True):
        st.session_state.playing = True
    if b.button("Pause", use_container_width=True):
        st.session_state.playing = False
    if st.button("Step", use_container_width=True):
        advance(scorer)
    if st.button("Reset", use_container_width=True):
        st.session_state.stream.reset()
        st.session_state.queue = QueueState()
        st.session_state.playing = False
        st.session_state.selected = None

    st.markdown('<div class="runline" style="margin-top:1.6rem">Decision layer</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p class="note">Score &nbsp; <b style="color:#C6CAD2">{info["selected_score"]}</b><br>'
        f'Calibration &nbsp; {info["calibration"]}<br>'
        f'Hold at &nbsp; {info["tier_boundaries"]["high"]} &nbsp;·&nbsp; '
        f'challenge at &nbsp; {info["tier_boundaries"]["medium"]}<br>'
        f'Features &nbsp; {info["features"]}</p>',
        unsafe_allow_html=True,
    )

stream: ReplayStream = st.session_state.stream
queue: QueueState = st.session_state.queue
summary = queue.summary()

st.markdown(
    '<div class="runline">IEEE-CIS &nbsp;&middot;&nbsp; held-out span &nbsp;&middot;&nbsp; '
    'replayed in arrival order</div>', unsafe_allow_html=True)
st.markdown('<h1 class="masthead">Transaction risk ledger</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="masthead-sub">{summary["processed"]:,} of '
    f'{len(stream.positions):,} entries posted</p>', unsafe_allow_html=True)
st.markdown(
    f'<div class="tape"><span style="width:{stream.progress * 100:.2f}%"></span></div>',
    unsafe_allow_html=True)

held = int((queue.decisions["tier"] == "high").sum()) if not queue.decisions.empty else 0
challenged = int((queue.decisions["tier"] == "medium").sum()) if not queue.decisions.empty else 0
fmt = lambda v, spec: "—" if not np.isfinite(v) else format(v, spec)  # noqa: E731

st.markdown(
    figure_line([
        ("Entries", f"{summary['processed']:,}", "posted", ""),
        ("Marked", f"{held:,}", "held for review", "pen"),
        ("Challenged", f"{challenged:,}", "step-up auth", ""),
        ("Confirmed", f"{summary['fraud_caught']:,}",
         f"of {summary['fraud_seen']:,} fraud present", "stamp"),
        ("Mark precision", fmt(summary["precision"], ".1%"), "marks that were fraud", ""),
        ("Throughput", fmt(summary["throughput_tps"], ",.0f"), "entries per second", ""),
    ]),
    unsafe_allow_html=True)

if queue.decisions.empty:
    st.markdown(
        '<div class="empty" style="margin-top:2rem"><p>The ledger is blank.</p>'
        "<p>Press Play to begin posting entries, or Step to post one batch.</p></div>",
        unsafe_allow_html=True)
    st.stop()

left, right = st.columns([7, 4], gap="large")

with left:
    st.markdown('<div class="section">Ledger &middot; held and challenged entries</div>',
                unsafe_allow_html=True)
    alerts = queue.alerts()
    if alerts.empty:
        st.markdown('<p class="note">No marks yet. Every entry so far cleared '
                    "without friction.</p>", unsafe_allow_html=True)
    else:
        view = alerts.head(60).reset_index(drop=True)
        table = pd.DataFrame({
            "": np.where(view["tier"] == "high", "▌", "·"),
            "Risk": view["risk_score"].astype(int),
            "Ruling": view["tier"].map(TIER_VERB),
            "Amount": view["amount"].astype(float).map("{:,.2f}".format),
            "Day": view["day"].astype(float).map("{:.2f}".format),
            "Outcome": np.where(view["is_fraud"] == 1, "■", "·"),
        })
        event = st.dataframe(
            ledger_style(table, view["tier"], view["is_fraud"] == 1),
            hide_index=True, use_container_width=True, height=430,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "": st.column_config.TextColumn(
                    "", width="small",
                    help="A mark in the margin means the entry was held for review."),
                "Risk": st.column_config.NumberColumn("Risk", width="small"),
                "Ruling": st.column_config.TextColumn("Ruling", width="small"),
                "Amount": st.column_config.TextColumn("Amount", width="small"),
                "Day": st.column_config.TextColumn("Day", width="small"),
                "Outcome": st.column_config.TextColumn(
                    "Was fraud", width="small",
                    help="Ground truth, stamped after the fact. Shown only because "
                         "this is an evaluation harness; a live ledger would not have it."),
            },
        )
        rows = event.selection.rows if event and event.selection else []
        st.session_state.selected = (
            int(view.iloc[rows[0]]["position"]) if rows else int(view.iloc[0]["position"]))
        st.markdown(
            '<p class="note">A <b>&#9612;</b> in the margin was held; a '
            "<b>&#9632;</b> turned out to be fraud. Strokes without stamps are "
            "false alarms. Select an entry to see what drove it.</p>",
            unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">Margin note</div>', unsafe_allow_html=True)
    position = st.session_state.selected
    if position is None:
        st.markdown('<p class="note">Select an entry from the ledger.</p>',
                    unsafe_allow_html=True)
    else:
        row = queue.decisions[queue.decisions["position"] == position].iloc[0]
        outcome = (
            '<span class="stamp">&#9632; confirmed fraud</span>' if row.is_fraud == 1
            else '<span style="color:#BFC9B8">&mdash; not fraud</span>')
        st.markdown(
            f'<div class="slip">Entry {int(row.position)} &nbsp;&middot;&nbsp; '
            f'<b>risk {int(row.risk_score)}</b> &nbsp;&middot;&nbsp; '
            f'<span class="pen">{TIER_VERB[row.tier]}</span><br>'
            f"amount {row.amount:,.2f} &nbsp;&middot;&nbsp; day {row.day:.2f} "
            f"&nbsp;&middot;&nbsp; p {row.p_calibrated:.4f}<br>{outcome}</div>",
            unsafe_allow_html=True)
        reasons = scorer.explain_position(position)
        if reasons is None:
            st.markdown('<p class="note">No tree model available for attribution.</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(plain_reasons(reasons["risk_factors"]), unsafe_allow_html=True)
            if reasons["mitigating_factors"]:
                st.markdown('<p class="note" style="margin:.4rem 0">Argued for clearing it</p>',
                            unsafe_allow_html=True)
                st.markdown(plain_reasons(reasons["mitigating_factors"][:3], against=True),
                            unsafe_allow_html=True)

st.markdown('<div class="section">Tally &middot; where the fraud sits</div>',
            unsafe_allow_html=True)
g1, g2, g3 = st.columns([3, 3, 3], gap="large")

with g1:
    decisions = queue.decisions
    edges = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 101)]
    total_fraud = max(int((decisions["is_fraud"] == 1).sum()), 1)
    bands = []
    for lo, hi in edges:
        part = decisions[(decisions["risk_score"] >= lo) & (decisions["risk_score"] < hi)]
        bands.append((f"{lo}-{hi - 1}", len(part) / len(decisions),
                      int((part["is_fraud"] == 1).sum()) / total_fraud))
    st.markdown(band_rows(bands), unsafe_allow_html=True)
    st.markdown(f'<p class="note" style="margin-top:.8rem">Alert rate '
                f'{summary["alert_rate"]:.2%} of volume.</p>', unsafe_allow_html=True)

with g2:
    tiers = queue.tier_counts()
    tiers["tier"] = tiers["tier"].map(TIER_VERB)
    st.dataframe(
        tiers.rename(columns={"tier": "Ruling", "transactions": "Entries",
                              "fraud": "Were fraud", "precision": "Precision"})
             .style.format({"Precision": "{:.1%}"}),
        hide_index=True, use_container_width=True)
    if queue.batch_latency_ms:
        latency = read_table("latency_product.csv")
        single = ""
        if latency is not None and "single_transaction" in set(latency["mode"]):
            r = latency.set_index("mode").loc["single_transaction"]
            single = (f" A single transaction through the online path takes "
                      f"<b>{r['p50_ms']:.1f} ms</b> p50, {r['p95_ms']:.1f} ms p95.")
        st.markdown(
            f'<p class="note" style="margin-top:.7rem">Median '
            f"<b>{np.median(queue.batch_latency_ms):.3f} ms</b> per entry, amortised "
            f"over batches of {batch_size}.{single}</p>", unsafe_allow_html=True)

with g3:
    st.markdown('<div class="runline">Score drift &middot; PSI vs calibration holdout</div>',
                unsafe_allow_html=True)
    reference = reference_scores()
    threshold = float(cfg.require("evaluation.psi_alert_threshold"))
    p = queue.decisions["p_calibrated"].to_numpy(np.float64)
    if reference is None or len(p) < DRIFT_WINDOW:
        st.markdown(f'<p class="note">The first window closes at {DRIFT_WINDOW:,} entries.</p>',
                    unsafe_allow_html=True)
    else:
        windows = []
        for k in range(len(p) // DRIFT_WINDOW):
            value = psi(reference, p[k * DRIFT_WINDOW:(k + 1) * DRIFT_WINDOW],
                        n_bins=int(cfg.require("evaluation.psi_bins")))
            windows.append({"Window": k + 1, "PSI": round(value, 4),
                            "Status": interpret(value, threshold)})
        frame = pd.DataFrame(windows)
        st.dataframe(frame.tail(8), hide_index=True, use_container_width=True)
        worst = frame["PSI"].max()
        if worst >= threshold:
            st.markdown(f'<p class="note"><b>PSI {worst:.3f} crossed {threshold:.2f}: '
                        "retraining recommended.</b></p>", unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="note">Worst window {worst:.4f}, below the '
                        f"{threshold:.2f} retraining threshold.</p>", unsafe_allow_html=True)

if st.session_state.playing and not stream.exhausted:
    advance(scorer)
    time.sleep(0.05)
    st.rerun()
elif stream.exhausted and summary["processed"]:
    st.markdown('<p class="note" style="margin-top:1.5rem">Replay complete — the '
                "whole held-out span has been scored.</p>", unsafe_allow_html=True)

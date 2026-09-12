"""Analyst console: the decision layer, watched while it runs.

    streamlit run serving/dashboard.py

This is a queue, not a dashboard. An analyst opens it to work a list in risk
order and to answer one question per row - why was this held? So the queue is
the hero, the reason codes sit beside it rather than a click away, and the
governance panel is a strip underneath rather than the main event.

Transactions arrive by replaying the held-out test span in chronological order,
which is the proxy for a live stream the report's assumptions describe. The
outcome column is shown because this is an evaluation harness rather than a
production console: a live system would not know it at decision time, and it is
here only so the queue can be judged against the truth.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "serving"))

from replay import QueueState, ReplayStream      # noqa: E402
from scorer import FraudScorer                    # noqa: E402
from theme import CSS, band_rows, figure_line, reason_rows  # noqa: E402

st.set_page_config(page_title="Fraud Risk Console", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

#: The queue speaks in verbs an analyst uses, and the same word is used from
#: tier to action to column heading so the vocabulary is learnable.
TIER_VERB = {"low": "Clear", "medium": "Challenge", "high": "Hold"}


@st.cache_resource(show_spinner="Loading decision layer...")
def get_scorer() -> FraudScorer:
    return FraudScorer()


def init_state(scorer: FraudScorer, batch_size: int) -> None:
    if "stream" not in st.session_state:
        st.session_state.stream = ReplayStream(scorer.test_positions(), batch_size)
        st.session_state.queue = QueueState()
        st.session_state.playing = False
        st.session_state.selected = None
    st.session_state.stream.batch_size = batch_size


def advance(scorer: FraudScorer) -> None:
    stream: ReplayStream = st.session_state.stream
    batch = stream.next_batch()
    if not len(batch):
        st.session_state.playing = False
        return
    started = time.perf_counter()
    decisions = scorer.score_positions(batch)
    st.session_state.queue.extend(decisions, (time.perf_counter() - started) * 1000.0)


def parse_reasons(raw: str) -> list[tuple[str, str, float]]:
    """Turn 'feature=value (+0.1234) | ...' into parts the bars can use."""
    out: list[tuple[str, str, float]] = []
    for chunk in str(raw).split(" | "):
        if "=" not in chunk or "(" not in chunk:
            continue
        feature, rest = chunk.split("=", 1)
        value, contribution = rest.rsplit("(", 1)
        try:
            amount = float(contribution.strip(" )+"))
        except ValueError:
            continue
        out.append((feature.strip(), value.strip(), amount))
    return out


# ------------------------------------------------------------------ setup --
scorer = get_scorer()
info = scorer.describe()

with st.sidebar:
    st.markdown('<div class="eyebrow">Replay</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="eyebrow" style="margin-top:1.6rem">Decision layer</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p class="note">Score &nbsp; <b style="color:#C6CAD2">{info["selected_score"]}</b><br>'
        f'Calibration &nbsp; {info["calibration"]}<br>'
        f'Hold at &nbsp; {info["tier_boundaries"]["high"]} &nbsp;·&nbsp; '
        f'challenge at &nbsp; {info["tier_boundaries"]["medium"]}<br>'
        f'Features &nbsp; {info["features"]}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="note" style="margin-top:1rem">One model is served, not six — '
        "the fusion stage chose it over the stack on held-out data.</p>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------- header --
stream: ReplayStream = st.session_state.stream
queue: QueueState = st.session_state.queue
summary = queue.summary()

st.markdown('<div class="eyebrow">IEEE-CIS · held-out span · days 141-183</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="masthead">Fraud risk console</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="masthead-sub">{summary["processed"]:,} of {len(stream.positions):,} '
    f"transactions replayed in arrival order</p>",
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="tape"><span style="width:{stream.progress * 100:.2f}%"></span></div>',
    unsafe_allow_html=True,
)

held = int((queue.decisions["tier"] == "high").sum()) if not queue.decisions.empty else 0
challenged = int((queue.decisions["tier"] == "medium").sum()) if not queue.decisions.empty else 0
fmt = lambda v, spec: "—" if not np.isfinite(v) else format(v, spec)  # noqa: E731

st.markdown(
    figure_line([
        ("Scanned", f"{summary['processed']:,}", None, False),
        ("Held", f"{held:,}", "manual review", True),
        ("Challenged", f"{challenged:,}", "step-up auth", False),
        ("Caught", f"{summary['fraud_caught']:,}",
         f"of {summary['fraud_seen']:,} fraud seen", True),
        ("Queue precision", fmt(summary["precision"], ".1%"), None, False),
        ("Throughput", fmt(summary["throughput_tps"], ",.0f"), "per second", False),
    ]),
    unsafe_allow_html=True,
)

if queue.decisions.empty:
    st.markdown(
        '<div class="empty" style="margin-top:2rem"><p>The queue is empty.</p>'
        "<p>Press Play to start the replay, or Step to advance one batch.</p></div>",
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------------ queue --
left, right = st.columns([3, 2], gap="large")

with left:
    st.markdown('<div class="section">Queue · riskiest first</div>', unsafe_allow_html=True)
    alerts = queue.alerts()
    if alerts.empty:
        st.markdown('<p class="note">Nothing flagged yet. Every transaction so far '
                    "cleared without friction.</p>", unsafe_allow_html=True)
    else:
        view = alerts.head(60).copy()
        table = pd.DataFrame({
            "Risk": view["risk_score"].astype(int),
            "Action": view["tier"].map(TIER_VERB),
            "Amount": view["amount"].astype(float),
            "Day": view["day"].astype(float),
            "Confirmed": np.where(view["is_fraud"] == 1, "fraud", ""),
            "Row": view["position"].astype(int),
        })
        event = st.dataframe(
            table, hide_index=True, use_container_width=True, height=430,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "Risk", min_value=0, max_value=100, format="%d", width="medium"),
                "Amount": st.column_config.NumberColumn("Amount", format="%.2f"),
                "Day": st.column_config.NumberColumn("Day", format="%.2f"),
                "Confirmed": st.column_config.TextColumn(
                    "Outcome", help="Ground truth, shown only because this is an "
                                    "evaluation harness. A live console would not have it."),
            },
        )
        rows = event.selection.rows if event and event.selection else []
        st.session_state.selected = (
            int(view.iloc[rows[0]]["position"]) if rows
            else int(view.iloc[0]["position"])
        )
        st.markdown('<p class="note">Select a row to see what drove its score.</p>',
                    unsafe_allow_html=True)

with right:
    st.markdown('<div class="section">Why it was flagged</div>', unsafe_allow_html=True)
    position = st.session_state.selected
    if position is None:
        st.markdown('<p class="note">Select an alert.</p>', unsafe_allow_html=True)
    else:
        row = queue.decisions[queue.decisions["position"] == position].iloc[0]
        outcome = ('<span class="true">confirmed fraud</span>' if row.is_fraud == 1
                   else '<span style="color:#4A525F">not fraud</span>')
        st.markdown(
            f'<div class="verdict">Row {int(row.position)} &nbsp;·&nbsp; '
            f'<b>risk {int(row.risk_score)}</b> &nbsp;·&nbsp; '
            f'<span class="held">{TIER_VERB[row.tier]}</span><br>'
            f"amount {row.amount:,.2f} &nbsp;·&nbsp; day {row.day:.2f} "
            f"&nbsp;·&nbsp; p {row.p_calibrated:.4f}<br>"
            f"outcome {outcome}</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Attributing..."):
            codes = scorer.explain_positions(np.array([position]))
        if codes.empty:
            st.markdown('<p class="note">No tree model available for attribution.</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(reason_rows(parse_reasons(codes.iloc[0]["reason_codes"])),
                        unsafe_allow_html=True)
            st.markdown(
                '<p class="note" style="margin-top:.9rem">Bars are SHAP '
                "contributions to this score. Only features that pushed it "
                "upward are listed: a reviewer asking why a transaction was "
                "held needs the evidence for the hold.</p>",
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------- governance --
st.markdown('<div class="section">Where the fraud sits</div>', unsafe_allow_html=True)
g1, g2, g3 = st.columns([2, 1, 1], gap="large")

with g1:
    decisions = queue.decisions
    edges = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 101)]
    total_fraud = max(int((decisions["is_fraud"] == 1).sum()), 1)
    bands = []
    for lo, hi in edges:
        part = decisions[(decisions["risk_score"] >= lo) & (decisions["risk_score"] < hi)]
        bands.append((
            f"{lo}-{hi - 1}",
            len(part) / len(decisions),
            int((part["is_fraud"] == 1).sum()) / total_fraud,
        ))
    st.markdown(band_rows(bands), unsafe_allow_html=True)
    st.markdown(
        f'<p class="note" style="margin-top:.8rem">Alert rate '
        f'{summary["alert_rate"]:.2%} of volume.</p>', unsafe_allow_html=True)

with g2:
    tiers = queue.tier_counts()
    tiers["tier"] = tiers["tier"].map(TIER_VERB)
    st.dataframe(
        tiers.rename(columns={"tier": "Action", "transactions": "Count",
                              "fraud": "Fraud", "precision": "Precision"})
             .style.format({"Precision": "{:.1%}"}),
        hide_index=True, use_container_width=True,
    )

with g3:
    if queue.batch_latency_ms:
        st.line_chart(
            pd.DataFrame({"ms per transaction": queue.batch_latency_ms}),
            height=150, color="#C9922E",
        )
        st.markdown(
            f'<p class="note">Median {np.median(queue.batch_latency_ms):.3f} ms '
            f"per transaction, amortised over batches of {batch_size}. "
            f"Scoring one at a time costs ~126 ms.</p>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------ auto-advance --
if st.session_state.playing and not stream.exhausted:
    advance(scorer)
    time.sleep(0.05)
    st.rerun()
elif stream.exhausted and summary["processed"]:
    st.markdown('<p class="note" style="margin-top:1.5rem">Replay complete — the '
                "whole held-out span has been scored.</p>", unsafe_allow_html=True)

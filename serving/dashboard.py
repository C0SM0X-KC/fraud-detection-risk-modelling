"""Analyst dashboard: the decision layer, watched while it runs.

    streamlit run serving/dashboard.py

Three things a fraud operation actually needs, on one screen. The queue, so an
analyst can see what to work and in what order. The reason codes behind any
alert, so an adverse decision can be explained and contested rather than
asserted. And the governance panel, so the people responsible for the model can
see its score distribution, its drift and its serving cost without asking for a
report.

Transactions arrive by replaying the held-out test span in chronological order,
which is the protocol section 3.3 describes. The outcome column is shown
because this is an evaluation harness, not a production console - a live system
would not know it at decision time, and it is displayed only so the queue can
be judged against the truth.
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

from replay import QueueState, ReplayStream          # noqa: E402
from scorer import FraudScorer                        # noqa: E402

st.set_page_config(
    page_title="Fraud Risk Console",
    page_icon="*",
    layout="wide",
)

TIER_COLOUR = {"low": "#2e7d32", "medium": "#ef6c00", "high": "#c62828"}


@st.cache_resource(show_spinner="Loading the decision layer...")
def get_scorer() -> FraudScorer:
    return FraudScorer()


def init_state(scorer: FraudScorer, batch_size: int) -> None:
    if "stream" not in st.session_state:
        st.session_state.stream = ReplayStream(scorer.test_positions(), batch_size)
        st.session_state.queue = QueueState()
        st.session_state.playing = False
        st.session_state.selected_alert = None
    st.session_state.stream.batch_size = batch_size


def advance(scorer: FraudScorer) -> None:
    stream: ReplayStream = st.session_state.stream
    batch = stream.next_batch()
    if not len(batch):
        st.session_state.playing = False
        return
    started = time.perf_counter()
    decisions = scorer.score_positions(batch)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    st.session_state.queue.extend(decisions, elapsed_ms)


# ----------------------------------------------------------------- sidebar --
scorer = get_scorer()
info = scorer.describe()

with st.sidebar:
    st.header("Replay controls")
    batch_size = st.select_slider(
        "Batch size", options=[64, 128, 256, 512, 1024], value=256,
        help="Single-transaction scoring costs ~126 ms; batching is what meets "
             "the real-time requirement, not a display convenience.",
    )
    init_state(scorer, batch_size)

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Play", use_container_width=True):
        st.session_state.playing = True
    if col_b.button("Pause", use_container_width=True):
        st.session_state.playing = False
    if col_c.button("Reset", use_container_width=True):
        st.session_state.stream.reset()
        st.session_state.queue = QueueState()
        st.session_state.playing = False
        st.session_state.selected_alert = None

    if st.button("Step one batch", use_container_width=True):
        advance(scorer)

    st.divider()
    st.caption("Decision layer")
    st.write(
        f"**Score:** {info['selected_score']}  \n"
        f"**Calibration:** {info['calibration']}  \n"
        f"**Tiers:** medium >= {info['tier_boundaries']['medium']}, "
        f"high >= {info['tier_boundaries']['high']}  \n"
        f"**Features:** {info['features']}"
    )
    st.caption(
        "One model is loaded rather than six because the fusion stage selected "
        "it over the stack on held-out data. Serving the rejected tiers would "
        "misrepresent the evaluated system."
    )

# ------------------------------------------------------------------ header --
st.title("Fraud detection and risk modelling - analyst console")
stream: ReplayStream = st.session_state.stream
queue: QueueState = st.session_state.queue
summary = queue.summary()

st.progress(stream.progress, text=f"Replayed {summary['processed']:,} of {len(stream.positions):,} transactions")

cols = st.columns(6)
cols[0].metric("Processed", f"{summary['processed']:,}")
cols[1].metric("Alerts raised", f"{summary['alerts']:,}")
cols[2].metric("Fraud caught", f"{summary['fraud_caught']:,}",
               delta=f"of {summary['fraud_seen']:,} seen")
cols[3].metric("Recall", "-" if not np.isfinite(summary["recall"]) else f"{summary['recall']:.1%}")
cols[4].metric("Alert precision",
               "-" if not np.isfinite(summary["precision"]) else f"{summary['precision']:.1%}")
cols[5].metric("Throughput",
               "-" if not np.isfinite(summary["throughput_tps"]) else f"{summary['throughput_tps']:,.0f}/s")

if queue.decisions.empty:
    st.info("Press **Play** or **Step one batch** in the sidebar to begin the replay.")
    st.stop()

# ------------------------------------------------------------------- queue --
left, right = st.columns([3, 2], gap="large")

with left:
    st.subheader("Alert queue")
    st.caption("Everything the policy did not auto-approve, riskiest first.")
    alerts = queue.alerts()
    if alerts.empty:
        st.write("No alerts yet.")
    else:
        view = alerts.head(40)[
            ["position", "day", "amount", "risk_score", "tier", "action", "is_fraud"]
        ].rename(columns={
            "position": "Row", "day": "Day", "amount": "Amount",
            "risk_score": "Risk", "tier": "Tier", "action": "Action",
            "is_fraud": "Actual fraud",
        })
        st.dataframe(
            view.style.map(
                lambda v: f"color: {TIER_COLOUR.get(v, 'inherit')}; font-weight: 600",
                subset=["Tier"],
            ),
            use_container_width=True, hide_index=True, height=420,
        )
        choice = st.selectbox(
            "Explain an alert", options=alerts["position"].head(40).tolist(),
            format_func=lambda p: f"row {p} - risk {int(alerts.loc[alerts.position == p, 'risk_score'].iloc[0])}",
        )
        st.session_state.selected_alert = choice

with right:
    st.subheader("Why was this flagged?")
    position = st.session_state.selected_alert
    if position is None:
        st.write("Select an alert to see its reason codes.")
    else:
        row = queue.decisions[queue.decisions["position"] == position].iloc[0]
        st.markdown(
            f"**Risk score {int(row.risk_score)}** &nbsp;·&nbsp; tier "
            f"`{row.tier}` &nbsp;·&nbsp; action `{row.action}`  \n"
            f"Calibrated probability {row.p_calibrated:.4f} &nbsp;·&nbsp; "
            f"amount {row.amount:,.2f} &nbsp;·&nbsp; day {row.day:.2f}"
        )
        with st.spinner("Computing SHAP attribution..."):
            codes = scorer.explain_positions(np.array([position]))
        if codes.empty:
            st.write("No tree model available for attribution.")
        else:
            reasons = codes.iloc[0]["reason_codes"]
            for item in str(reasons).split(" | "):
                st.markdown(f"- {item}")
            st.caption(
                "Each line is a feature that pushed this score upward, with its "
                "value and its SHAP contribution. Only positive contributions "
                "are listed: a reviewer asking why a transaction was held needs "
                "the evidence for the hold."
            )

# ------------------------------------------------------------- governance --
st.divider()
st.subheader("Governance")
gov_left, gov_mid, gov_right = st.columns(3, gap="large")

with gov_left:
    st.caption("Risk-score distribution")
    hist = (
        queue.decisions.assign(band=(queue.decisions["risk_score"] // 5) * 5)
        .groupby("band").size().rename("transactions").reset_index()
    )
    st.bar_chart(hist, x="band", y="transactions", height=240)

with gov_mid:
    st.caption("Tier breakdown")
    tiers = queue.tier_counts()
    st.dataframe(
        tiers.style.format({"precision": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Alert rate {summary['alert_rate']:.2%} of volume."
        if np.isfinite(summary["alert_rate"]) else ""
    )

with gov_right:
    st.caption("Per-transaction latency, by batch")
    if queue.batch_latency_ms:
        lat = pd.DataFrame({
            "batch": np.arange(1, len(queue.batch_latency_ms) + 1),
            "ms_per_txn": queue.batch_latency_ms,
        })
        st.line_chart(lat, x="batch", y="ms_per_txn", height=240)
        st.caption(
            f"Median {np.median(queue.batch_latency_ms):.3f} ms/transaction "
            f"amortised over batches of {batch_size}."
        )

# ------------------------------------------------------------ auto-advance --
if st.session_state.playing and not stream.exhausted:
    advance(scorer)
    time.sleep(0.05)
    st.rerun()
elif stream.exhausted and summary["processed"]:
    st.success("Replay complete - the whole held-out span has been scored.")

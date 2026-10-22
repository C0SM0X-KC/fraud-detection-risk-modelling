"""Check a transaction: the product's front door."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from console import get_service
from theme import group_rows, plain_reasons, verdict_card

service = get_service()
store = service.store
info = service.describe()

EDITABLE = [
    ("TransactionAmt", "Amount", "number"),
    ("ProductCD", "Product code", "choice"),
    ("card1", "Card identifier (card1)", "int"),
    ("card4", "Card network", "choice"),
    ("card6", "Card type", "choice"),
    ("addr1", "Billing region (addr1)", "number"),
    ("P_emaildomain", "Purchaser e-mail domain", "choice"),
    ("R_emaildomain", "Recipient e-mail domain", "choice"),
    ("DeviceType", "Device type", "choice"),
    ("DeviceInfo", "Device model", "text"),
    ("dist1", "Distance between addresses (dist1)", "number"),
]


def choices(column: str) -> list[str]:
    levels = service.categories.get(column, [])
    return [""] + [c for c in levels if c not in ("__rare__", "__missing__")]


if "checks" not in st.session_state:
    st.session_state.checks = []          # [(raw, decision)]

st.markdown('<div class="runline">Product &nbsp;&middot;&nbsp; online scoring '
            '&nbsp;&middot;&nbsp; features computed from live entity history</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="masthead">Check a transaction</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="masthead-sub">Model <b>{info["model"]}</b> &middot; '
    f'{info["history_rows"]:,} transactions of history &middot; '
    f'{info["transactions_scored_online"]:,} scored online this session</p>',
    unsafe_allow_html=True)

templates = service.templates()
if not templates:
    st.error("No example transactions stored. Re-run `python -m frds features`.")
    st.stop()

frauds = [t for t in templates if t["was_fraud"]][:15]
legits = [t for t in templates if not t["was_fraud"]][:15]
examples = [t for pair in zip(frauds, legits) for t in pair]
labels = [
    f"Example {i + 1} — {'fraud' if t['was_fraud'] else 'legitimate'} in the data, "
    f"amount {t['fields'].get('TransactionAmt', 0):,.2f}"
    for i, t in enumerate(examples)
]

left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown('<div class="section">Transaction</div>', unsafe_allow_html=True)
    pick = st.selectbox("Start from a real transaction", range(len(examples)),
                        format_func=lambda i: labels[i])
    template = examples[pick]["fields"]

    with st.form("transaction"):
        edited = {}
        cols = st.columns(2)
        for k, (field, label, kind) in enumerate(EDITABLE):
            current = template.get(field)
            with cols[k % 2]:
                key = f"{field}_{pick}"
                if kind == "number":
                    edited[field] = st.number_input(
                        label, value=None if current is None else float(current),
                        format="%.2f", key=key)
                elif kind == "int":
                    edited[field] = st.number_input(
                        label, value=int(current) if current is not None else 10000,
                        step=1, key=key)
                elif kind == "choice":
                    options = choices(field)
                    value = "" if current is None else str(current)
                    if value not in options:
                        options = options + [value]
                    edited[field] = st.selectbox(label, options, index=options.index(value),
                                                 key=key)
                else:
                    edited[field] = st.text_input(label, value="" if current is None else str(current),
                                                  key=key)
        gap = st.number_input("Seconds after the latest transaction seen", min_value=0,
                              value=60, step=30)
        remember = st.checkbox("Remember this transaction (append to entity history)", value=True)
        submitted = st.form_submit_button("Score transaction", use_container_width=True)

    st.markdown(
        f'<p class="note">The other {len(template) - len(EDITABLE)} fields '
        "(Vesta C, D, M, V and identity columns) are carried over from the example. "
        "An empty field is sent as missing, as in the benchmark.</p>",
        unsafe_allow_html=True)

    def build_raw(offset: int) -> dict:
        raw = {k: v for k, v in template.items() if v is not None}
        for field, value in edited.items():
            if value is None or value == "":
                raw.pop(field, None)
            else:
                raw[field] = int(value) if field == "card1" else value
        raw[store.time_col] = store.latest_time + int(offset)
        return raw

    if submitted:
        raw = build_raw(gap)
        try:
            decision = service.score(raw, remember=remember)
            st.session_state.checks.append((raw, decision))
        except ValueError as exc:
            st.error(str(exc))

with right:
    st.markdown('<div class="section">Decision</div>', unsafe_allow_html=True)
    if not st.session_state.checks:
        st.markdown('<div class="empty"><p>No transaction scored yet.</p>'
                    "<p>Pick an example, change what you like, and press Score.</p></div>",
                    unsafe_allow_html=True)
        st.stop()

    raw, decision = st.session_state.checks[-1]
    st.markdown(verdict_card(decision, info["tier_boundaries"]["medium"],
                             info["tier_boundaries"]["high"]), unsafe_allow_html=True)

    history = decision["history"]
    seen = ", ".join(
        f'{name} {"seen " + str(h["prior_transactions"]) + "x" if h["known"] else "new"}'
        for name, h in history.items())
    t = decision["timings_ms"]
    st.markdown(
        f'<p class="note">History &nbsp; {seen}<br>'
        f'Timing &nbsp; features {t["features_ms"]:.1f} ms &middot; model {t["model_ms"]:.2f} ms '
        f'&middot; decision {t["decision_ms"]:.2f} ms &middot; reasons '
        f'{t.get("explanation_ms", 0):.1f} ms</p>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    if a.button("Send again in 30 s", use_container_width=True,
                help="Same card, same details, 30 seconds later: watch velocity build"):
        again = dict(raw)
        again.pop(store.id_col, None)
        again[store.time_col] = store.latest_time + 30
        st.session_state.checks.append((again, service.score(again)))
        st.rerun()
    if b.button("Confirm fraud", use_container_width=True):
        service.record_outcome(decision["transaction_id"], True)
        st.toast(f"Recorded transaction {decision['transaction_id']} as fraud; "
                 "the card's and account's history now include it.")
    if c.button("Confirm legitimate", use_container_width=True):
        service.record_outcome(decision["transaction_id"], False)
        st.toast(f"Recorded transaction {decision['transaction_id']} as legitimate.")

    reasons = decision.get("reasons")
    if reasons:
        r1, r2 = st.columns(2, gap="medium")
        with r1:
            st.markdown('<div class="section">Raised the risk</div>', unsafe_allow_html=True)
            st.markdown(plain_reasons(reasons["risk_factors"]), unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="section">Lowered the risk</div>', unsafe_allow_html=True)
            st.markdown(plain_reasons(reasons["mitigating_factors"], against=True),
                        unsafe_allow_html=True)
        st.markdown('<div class="section">Evidence by family</div>', unsafe_allow_html=True)
        st.markdown(group_rows(reasons["evidence_by_group"]), unsafe_allow_html=True)
        st.markdown(
            '<p class="note" style="margin-top:.6rem">Each figure is an exact TreeSHAP '
            f'contribution to the {reasons["explained_model"]} model\'s log-odds, before '
            "calibration; together with the base value they sum to the model's output. "
            "Calibration is monotone, so these explain why the transaction ranks where it "
            "does.</p>", unsafe_allow_html=True)

st.markdown('<div class="section">This session</div>', unsafe_allow_html=True)
rows = [{
    "Transaction": d["transaction_id"],
    "Amount": f'{d["amount"]:,.2f}',
    "Card": r.get("card1"),
    "Risk": d["risk_score"],
    "Verdict": d["verdict"],
    "Action": d["action"],
    "ms": d["timings_ms"]["total_ms"],
} for r, d in reversed(st.session_state.checks)]
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

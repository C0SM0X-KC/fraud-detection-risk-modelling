"""Fraud analytics: who, when and how much, with filters, tests and segment performance."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from console import figure, read_table
from frds.config import load_config
from theme import TOKENS, figure_line

ORDERS = {
    "amount_band": ["under 25", "25-50", "50-100", "100-250", "250-500", "500+"],
    "account_history": ["first seen", "1-4 earlier", "5-19 earlier", "20+ earlier"],
    "split": ["train", "validation", "test"],
}
DIMENSIONS = {
    "Product code": "ProductCD",
    "Card network": "card4",
    "Card type": "card6",
    "Device type": "DeviceType",
    "Purchaser e-mail domain": "email",
    "Identity record present": "has_identity",
    "Amount band": "amount_band",
    "Account history": "account_history",
    "Hour of day": "hour",
}


def section(title: str) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)


def styled(chart: alt.Chart) -> alt.Chart:
    return chart.configure(background="transparent").configure_view(strokeOpacity=0) \
        .configure_axis(labelColor=TOKENS["ink"], titleColor=TOKENS["faint"],
                        gridColor=TOKENS["rule"], domainColor=TOKENS["rule"],
                        tickColor=TOKENS["rule"], labelFont="Azeret Mono", titleFont="Archivo") \
        .configure_legend(labelColor=TOKENS["ink"], titleColor=TOKENS["faint"])


@st.cache_data(show_spinner="Loading the transaction table...")
def transactions() -> pd.DataFrame | None:
    path = load_config().path("processed") / "analytics_transactions.parquet"
    return pd.read_parquet(path) if path.exists() else None


def options(frame: pd.DataFrame, column: str) -> list[str]:
    present = [str(v) for v in frame[column].dropna().unique()]
    order = ORDERS.get(column)
    return [v for v in order if v in present] if order else sorted(present)


def rates(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = frame.groupby(column, observed=True).agg(
        transactions=("isFraud", "size"), fraud=("isFraud", "sum"),
        value=("TransactionAmt", "sum")).reset_index()
    grouped["fraud_rate"] = grouped["fraud"] / grouped["transactions"]
    grouped["share_of_volume"] = grouped["transactions"] / max(len(frame), 1)
    grouped[column] = grouped[column].astype(str)
    return grouped


st.markdown('<div class="runline">Data analytics &nbsp;&middot;&nbsp; IEEE-CIS '
            '&nbsp;&middot;&nbsp; descriptive, diagnostic and segment views</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="masthead">Fraud analytics</h1>', unsafe_allow_html=True)

data = transactions()
if data is None:
    st.markdown('<div class="empty"><p>No analytics table yet.</p>'
                "<p>Run python -m frds analytics.</p></div>", unsafe_allow_html=True)
    st.stop()

with st.sidebar:
    st.markdown('<div class="runline">Filters</div>', unsafe_allow_html=True)
    spans = st.multiselect("Period", options(data, "split"), default=["train"],
                           help="train = days 1-121, validation = the next 20 days, "
                                "test = the last 42 days (model results are reported here)")
    chosen = {}
    for label in ("Product code", "Card network", "Card type", "Device type",
                  "Purchaser e-mail domain", "Amount band", "Account history"):
        column = DIMENSIONS[label]
        chosen[column] = st.multiselect(label, options(data, column), placeholder="All")

mask = data["split"].isin(spans or options(data, "split"))
for column, picked in chosen.items():
    if picked:
        mask &= data[column].astype(str).isin(picked)
view = data[mask]

st.markdown(
    f'<p class="masthead-sub">{len(view):,} of {len(data):,} transactions selected '
    f'&middot; period {", ".join(spans) if spans else "all"}</p>', unsafe_allow_html=True)
if view.empty:
    st.markdown('<div class="empty"><p>No transactions match these filters.</p></div>',
                unsafe_allow_html=True)
    st.stop()

fraud = view[view["isFraud"] == 1]
legit = view[view["isFraud"] == 0]
st.markdown(figure_line([
    ("Transactions", f"{len(view):,}", f"{view['day'].nunique()} days", ""),
    ("Fraud cases", f"{len(fraud):,}", None, "stamp"),
    ("Fraud rate", f"{len(fraud) / len(view):.2%}",
     f"overall {data['isFraud'].mean():.2%}", "pen"),
    ("Fraud value", f"{fraud['TransactionAmt'].sum():,.0f}",
     f"{fraud['TransactionAmt'].sum() / max(view['TransactionAmt'].sum(), 1):.1%} of value", ""),
    ("Median amount", f"{legit['TransactionAmt'].median():,.0f} / "
                      f"{fraud['TransactionAmt'].median() if len(fraud) else 0:,.0f}",
     "legitimate / fraud", ""),
]), unsafe_allow_html=True)

scored = view[view["risk_score"] >= 0]
if len(scored):
    flagged = scored["tier"].astype(str) != "low"
    held = scored["tier"].astype(str) == "high"
    is_fraud = scored["isFraud"] == 1
    st.markdown(
        '<div class="runline" style="margin-top:1rem">Decision layer on the selected '
        'test-span transactions</div>', unsafe_allow_html=True)
    st.markdown(figure_line([
        ("Alerts raised", f"{int(flagged.sum()):,}", f"{flagged.mean():.1%} of volume", ""),
        ("Held for review", f"{int(held.sum()):,}", "risk 70-100", "pen"),
        ("Fraud caught", f"{flagged[is_fraud].mean():.1%}" if is_fraud.any() else "-",
         "challenged or held", "pen"),
        ("Precision of holds", f"{is_fraud[held].mean():.1%}" if held.any() else "-",
         "held that were fraud", ""),
        ("False-positive rate", f"{flagged[~is_fraud].mean():.2%}" if (~is_fraud).any() else "-",
         "legitimate challenged or held", ""),
    ]), unsafe_allow_html=True)

section("Trend · daily volume and fraud rate")
daily = view.groupby("day").agg(transactions=("isFraud", "size"),
                                fraud=("isFraud", "sum")).reset_index()
daily["fraud_rate"] = daily["fraud"] / daily["transactions"]
base = alt.Chart(daily).encode(x=alt.X("day:Q", title="Day of the dataset"))
bars = base.mark_bar(color=TOKENS["rule"], opacity=0.9).encode(
    y=alt.Y("transactions:Q", title="Transactions per day"),
    tooltip=["day", "transactions", "fraud", alt.Tooltip("fraud_rate:Q", format=".2%")])
line = base.mark_line(color=TOKENS["pen"], strokeWidth=1.8).encode(
    y=alt.Y("fraud_rate:Q", title="Fraud rate", axis=alt.Axis(format="%")))
st.altair_chart(styled(alt.layer(bars, line).resolve_scale(y="independent")
                       .properties(height=260)), use_container_width=True, theme=None)

section("Breakdown · fraud rate by segment")
left, right = st.columns([3, 2], gap="large")
with left:
    label = st.selectbox("Break down by", list(DIMENSIONS), index=0)
    column = DIMENSIONS[label]
    table = rates(view, column)
    order = ORDERS.get(column) or (list(table[column]) if column == "hour"
                                   else list(table.sort_values("fraud_rate")[column])[::-1])
    chart = alt.Chart(table).mark_bar(color=TOKENS["pen"]).encode(
        x=alt.X(f"{column}:N", sort=order, title=label),
        y=alt.Y("fraud_rate:Q", title="Fraud rate", axis=alt.Axis(format="%")),
        tooltip=[column, "transactions", "fraud",
                 alt.Tooltip("fraud_rate:Q", format=".2%"),
                 alt.Tooltip("share_of_volume:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"overall": [view["isFraud"].mean()]})).mark_rule(
        color=TOKENS["faint"], strokeDash=[4, 4]).encode(y="overall:Q")
    st.altair_chart(styled((chart + rule).properties(height=300)),
                    use_container_width=True, theme=None)
with right:
    note("Dashed line: fraud rate of the whole selection. A segment far above it is "
         "riskier than average; its share of volume says how much it matters.")
    st.dataframe(table.rename(columns={column: label}).style.format({
        "transactions": "{:,}", "fraud": "{:,}", "value": "{:,.0f}",
        "fraud_rate": "{:.2%}", "share_of_volume": "{:.1%}"}),
        hide_index=True, use_container_width=True)

section("Pareto · how concentrated fraud value is across accounts")
summary_path = load_config().path("reports") / "metrics" / "analytics_summary.json"
p1, p2 = st.columns(2, gap="large")
with p1:
    path = figure("eda_pareto.png")
    if path:
        st.image(str(path), use_container_width=True)
with p2:
    if summary_path.exists():
        import json

        s = json.loads(summary_path.read_text())
        c = s["concentration"]
        note(f"Training span: <b>{c['accounts_with_fraud']:,}</b> of {c['accounts']:,} accounts "
             f"(<b>{c['share_of_accounts_with_fraud']:.2%}</b>) had any fraud. "
             f"<b>{c['fraud_accounts_for_80pct_value']:.0%}</b> of those accounts carry 80% of "
             f"the fraud value, and <b>{c['fraud_on_account_with_earlier_fraud']:.0%}</b> of "
             "fraud transactions came from an account that had already been used for fraud. "
             "That repetition is why an account's fraud history is the strongest evidence the "
             "model uses.")

section("Model by segment · where the system works well and where it struggles")
segments = read_table("segment_performance.csv")
if segments is None:
    note("Not produced yet. Run <b>python -m frds analytics</b>.")
else:
    pick = st.selectbox("Segment by", list(segments["dimension"].unique()))
    st.dataframe(segments[segments["dimension"] == pick].drop(columns="dimension").style.format({
        "transactions": "{:,}", "share_of_volume": "{:.1%}", "fraud": "{:,}",
        "fraud_rate": "{:.2%}", "auc_pr": "{:.3f}", "fraud_caught_high": "{:.1%}",
        "precision_high": "{:.1%}", "fraud_caught_flagged": "{:.1%}",
        "false_positive_rate": "{:.2%}", "fraud_value_caught": "{:.1%}"}, na_rep="—"),
        hide_index=True, use_container_width=True)
    note("Held-out test span, decision layer with the 0-29 / 30-69 / 70-100 bands. "
         "AUC-PR is shown only for segments with at least 20 fraud cases.")

section("Statistics · are the differences real, and how sure are the results")
s1, s2 = st.columns(2, gap="large")
with s1:
    note("Chi-square test of each category against fraud. Cramér's V is the effect size: "
         "under 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, above 0.5 large.")
    cat = read_table("stats_categorical_tests.csv")
    if cat is not None:
        st.dataframe(cat[["variable", "levels", "chi2", "p_bonferroni", "cramers_v", "strength"]]
                     .style.format({"chi2": "{:,.0f}", "p_bonferroni": "{:.1e}",
                                    "cramers_v": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
with s2:
    note("Mann-Whitney U test of each number, fraud against legitimate. "
         "<b>prob_fraud_higher</b> is the chance a random fraud has the larger value "
         "(0.5 = no difference); rank-biserial r is the same effect on a -1 to 1 scale.")
    num = read_table("stats_numeric_tests.csv")
    if num is not None:
        st.dataframe(num[["variable", "median_legitimate", "median_fraud", "p_bonferroni",
                          "prob_fraud_higher", "rank_biserial", "strength"]]
                     .style.format({"median_legitimate": "{:,.1f}", "median_fraud": "{:,.1f}",
                                    "p_bonferroni": "{:.1e}", "prob_fraud_higher": "{:.3f}",
                                    "rank_biserial": "{:+.3f}"}),
                     hide_index=True, use_container_width=True)
note("95% confidence intervals for the test-span results, from 1,000 resamples of whole days.")
boot = read_table("stats_bootstrap_ci.csv")
if boot is not None:
    st.dataframe(boot.style.format({"estimate": "{:.4f}", "ci_low": "{:.4f}",
                                    "ci_high": "{:.4f}"}),
                 hide_index=True, use_container_width=True)

section("Exploratory figures · training span")
names = ["eda_timeline.png", "eda_categories.png", "eda_amount.png", "eda_time_cycles.png",
         "eda_missingness.png", "eda_correlation.png", "stats_effect_sizes.png",
         "segment_detection.png"]
present = [figure(n) for n in names if figure(n)]
cols = st.columns(2)
for k, path in enumerate(present):
    cols[k % 2].image(str(path), use_container_width=True)

"""Governance layer (L5): reason codes an analyst can read."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ENTITY = {
    "uid": "account",
    "card": "card",
    "device": "device",
    "address": "billing address",
    "email": "purchaser e-mail domain",
    "merchant": "merchant (product and region)",
}

RAW = {
    "TransactionAmt": "Transaction amount",
    "ProductCD": "Product code",
    "card1": "Card identifier",
    "card2": "Card attribute card2",
    "card3": "Card attribute card3",
    "card4": "Card network",
    "card5": "Card attribute card5",
    "card6": "Card type",
    "addr1": "Billing region",
    "addr2": "Billing country",
    "dist1": "Distance between addresses (dist1)",
    "dist2": "Distance between addresses (dist2)",
    "P_emaildomain": "Purchaser e-mail domain",
    "R_emaildomain": "Recipient e-mail domain",
    "DeviceType": "Device type",
    "DeviceInfo": "Device model",
}

#: Evidence families, in the order an analyst would read them.
GROUPS = (
    "Entity fraud history",
    "Transaction velocity",
    "Recency and spending pattern",
    "Deviation from account norms",
    "Linked entities (graph)",
    "Amount and timing",
    "Identity and consistency checks",
    "Categorical profile",
    "Card and address attributes",
    "Vesta counts (C, masked)",
    "Vesta time deltas (D, masked)",
    "Vesta engineered features (V, masked)",
    "Identity fields (id, masked)",
)


def feature_group(name: str) -> str:
    if name.endswith("_prior_fraud_rate"):
        return "Entity fraud history"
    if name.startswith("vel_"):
        return "Transaction velocity"
    if name.startswith(("recency_", "roll_")):
        return "Recency and spending pattern"
    if name.startswith(("agg_", "freq_uid_")):
        return "Deviation from account norms"
    if name.startswith("graph_"):
        return "Linked entities (graph)"
    if name.startswith(("txn_", "amt_")) or name == "TransactionAmt":
        return "Amount and timing"
    if name.startswith(("mismatch_", "missing_", "match_flag_")) or name == "null_count_row":
        return "Identity and consistency checks"
    if re.fullmatch(r"M\d+", name):
        return "Identity and consistency checks"
    if name.startswith(("freq_", "cat_")):
        return "Categorical profile"
    if name.startswith(("card", "addr", "dist")):
        return "Card and address attributes"
    if re.fullmatch(r"C\d+", name):
        return "Vesta counts (C, masked)"
    if re.fullmatch(r"D\d+", name):
        return "Vesta time deltas (D, masked)"
    if re.fullmatch(r"V\d+", name):
        return "Vesta engineered features (V, masked)"
    if name.startswith("id_"):
        return "Identity fields (id, masked)"
    return "Categorical profile"


# ------------------------------------------------------------ formatting ----
def _num(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "missing"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.{digits}f}"


def _pct(v: float | None, digits: int = 0) -> str:
    return "an unknown share" if v is None else f"{v:.{digits}%}"


def _duration(seconds: float) -> str:
    s = float(seconds)
    if s < 120:
        return f"{s:.0f} seconds"
    if s < 7200:
        return f"{s / 60:.0f} minutes"
    if s < 172_800:
        return f"{s / 3600:.1f} hours"
    return f"{s / 86_400:.1f} days"


def _window(hours: int) -> str:
    return {1: "hour", 24: "24 hours", 168: "7 days"}.get(hours, f"{hours} hours")


def _missing(v) -> bool:
    return v is None or (isinstance(v, (float, np.floating)) and np.isnan(v))


# ------------------------------------------------------------- sentences ----
def describe(name: str, value, raw: dict | None = None,
             categories: dict[str, list[str]] | None = None) -> str:
    """One sentence stating what ``name = value`` means for this transaction."""
    raw = raw or {}
    v = None if _missing(value) else float(value)

    m = re.fullmatch(r"graph_(\w+?)_prior_fraud_rate", name)
    if m:
        who = ENTITY.get(m.group(1), m.group(1))
        return f"Smoothed fraud rate of this {who}'s earlier transactions is {_pct(v, 1)}"

    m = re.fullmatch(r"vel_(\w+?)_cnt_(\d+)h", name)
    if m:
        who, hours = ENTITY.get(m.group(1), m.group(1)), int(m.group(2))
        n = int(v or 0)
        return f"{n} earlier transaction{'s' if n != 1 else ''} on this {who} in the last {_window(hours)}"
    m = re.fullmatch(r"vel_(\w+?)_amt_(\d+)h", name)
    if m:
        who, hours = ENTITY.get(m.group(1), m.group(1)), int(m.group(2))
        return f"{_num(v)} already spent on this {who} in the last {_window(hours)}"
    m = re.fullmatch(r"vel_(\w+?)_rate_per_day", name)
    if m:
        return f"This {ENTITY.get(m.group(1), m.group(1))} transacts {_num(v)} times per day on average"
    m = re.fullmatch(r"vel_(\w+?)_age_days", name)
    if m:
        who = ENTITY.get(m.group(1), m.group(1))
        return (f"First transaction seen for this {who}" if not v
                else f"This {who} was first seen {_num(v, 1)} days ago")
    m = re.fullmatch(r"vel_(\w+?)_cnt_total", name)
    if m:
        return f"{_num(v)} earlier transactions on this {ENTITY.get(m.group(1), m.group(1))}"

    m = re.fullmatch(r"recency_(\w+?)_seconds", name)
    if m:
        who = ENTITY.get(m.group(1), m.group(1))
        return (f"No earlier transaction on this {who}" if v is None
                else f"{_duration(v)} since this {who}'s previous transaction")

    m = re.fullmatch(r"roll_(\w+?)_amt_(mean|std|z|ratio)", name)
    if m:
        who, kind = ENTITY.get(m.group(1), m.group(1)), m.group(2)
        if v is None:
            return f"No spending history on this {who} to compare against"
        return {
            "mean": f"This {who}'s average transaction amount is {_num(v)}",
            "std": f"This {who}'s transaction amounts vary by {_num(v)} (standard deviation)",
            "z": f"Amount is {v:+.1f} standard deviations from this {who}'s usual",
            "ratio": f"Amount is {v:.1f}x this {who}'s average",
        }[kind]

    m = re.fullmatch(r"graph_(\w+?)_distinct_(\w+?)_share", name)
    if m:
        left, right = ENTITY.get(m.group(1), m.group(1)), ENTITY.get(m.group(2), m.group(2))
        return f"{_pct(v).capitalize()} of this {left}'s transactions brought a {right} it had not used before"
    m = re.fullmatch(r"graph_(\w+?)_distinct_(\w+)", name)
    if m:
        left, right = ENTITY.get(m.group(1), m.group(1)), ENTITY.get(m.group(2), m.group(2))
        return f"This {left} has been used with {_num(v)} different {right}s"
    m = re.fullmatch(r"graph_(\w+?)_degree_per_day", name)
    if m:
        return f"This {ENTITY.get(m.group(1), m.group(1))} appears in {_num(v)} transactions per day"
    m = re.fullmatch(r"graph_(\w+?)_degree", name)
    if m:
        return f"This {ENTITY.get(m.group(1), m.group(1))} appears in {_num(v)} earlier transactions"
    if name == "graph_card_device_spread":
        return f"This card's device pool is {_pct(v)} of its transaction count"

    m = re.fullmatch(r"agg_(\w+?)_(\w+)_(mean|ratio)", name)
    if m:
        who, col, kind = ENTITY.get(m.group(1), m.group(1)), m.group(2), m.group(3)
        label = RAW.get(col, f"{col} (masked)")
        if v is None:
            return f"No history of {label} on this {who}"
        if kind == "ratio":
            return f"{label} is {v:.1f}x this {who}'s average"
        return f"This {who}'s average {label} is {_num(v)}"

    if name.startswith("freq_uid_"):
        return f"This card-and-address combination appeared {_num(v)} times in training"

    if name == "amt_log":
        return f"Transaction amount is {_num(math.expm1(v))}"
    if name == "amt_cents":
        return f"Amount ends in .{int(round((v or 0) * 100)):02d}"
    if name == "amt_is_round":
        return "Amount is a round number" if v else "Amount is not a round number"
    if name == "amt_decimal_len":
        return f"Amount is recorded with {_num(v)} decimal digits"
    if name == "txn_hour":
        return f"Made at hour {_num(v)}:00 of the dataset clock"
    if name == "txn_dow":
        return f"Made on day {_num(v)} of the dataset week"

    fixed = {
        "mismatch_email_domain": ("Purchaser and recipient e-mail domains differ",
                                  "Purchaser and recipient e-mail domains agree"),
        "mismatch_email_suffix": ("Purchaser and recipient e-mail suffixes differ",
                                  "Purchaser and recipient e-mail suffixes agree"),
        "missing_r_email": ("No recipient e-mail supplied", "Recipient e-mail supplied"),
        "missing_geo": ("No address distance recorded", "Address distance recorded"),
        "missing_device": ("No device information supplied", "Device information supplied"),
    }
    if name in fixed:
        return fixed[name][0] if v else fixed[name][1]
    if name == "mismatch_geo_distance":
        return ("No distance between addresses recorded" if v is None
                else f"Distance between billing and other address is {_num(v)}")
    if name == "match_flag_true_count":
        return f"{_num(v)} of the M match checks passed"
    if name == "match_flag_missing_count":
        return f"{_num(v)} of the M match checks were not performed"
    if name == "null_count_row":
        return f"{_num(v)} of the transaction's fields are empty"

    m = re.fullmatch(r"cat_(\w+)", name)
    if m:
        col = m.group(1)
        label = RAW.get(col, f"{col} (masked)")
        shown = raw.get(col)
        if _missing(shown) and categories and col in categories and v is not None and v >= 0:
            shown = categories[col][int(v)]
        if _missing(shown) or shown == "__missing__":
            return f"{label} not supplied"
        if shown == "__rare__":
            return f"{label} is a value rarely seen in training"
        return f"{label} is {shown}"
    m = re.fullmatch(r"freq_(\w+)", name)
    if m:
        label = RAW.get(m.group(1), f"{m.group(1)} (masked)")
        if not v:
            return f"{label} value never or rarely seen in training"
        return f"{label} value seen {_num(v)} times in training"

    if name in RAW:
        return f"{RAW[name]} is {_num(v) if v is not None else 'missing'}"
    if re.fullmatch(r"D\d+", name):
        return f"Vesta time feature {name} (meaning masked) is {_num(v)}"
    if re.fullmatch(r"[CVM]\d+", name) or name.startswith("id_"):
        return f"Masked provider feature {name} is {_num(v)}"
    return f"{name} is {_num(v)}"


# ------------------------------------------------------------- assembly ----
@dataclass
class Reason:
    feature: str
    group: str
    value: float | None
    contribution: float
    text: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["contribution"] = round(self.contribution, 4)
        if d["value"] is not None:
            d["value"] = round(float(d["value"]), 6)
        return d


def explain_row(
    contributions: np.ndarray,
    features: pd.Series,
    raw: dict | None = None,
    categories: dict[str, list[str]] | None = None,
    top_k: int = 5,
) -> dict:
    """Reasons for and against one score, and the evidence by family."""
    names = list(features.index)
    contributions = np.asarray(contributions, dtype=np.float64)
    order = np.argsort(contributions)

    def reason(j: int) -> Reason:
        value = features.iloc[j]
        value = None if _missing(value) else float(value)
        return Reason(names[j], feature_group(names[j]), value, float(contributions[j]),
                      describe(names[j], value, raw, categories))

    risk = [reason(j) for j in order[::-1][:top_k] if contributions[j] > 0]
    mitigating = [reason(j) for j in order[:top_k] if contributions[j] < 0]

    by_group: dict[str, float] = {}
    for name, c in zip(names, contributions):
        g = feature_group(name)
        by_group[g] = by_group.get(g, 0.0) + float(c)
    groups = sorted(by_group.items(), key=lambda kv: -kv[1])
    return {
        "risk_factors": [r.to_dict() for r in risk],
        "mitigating_factors": [r.to_dict() for r in mitigating],
        "evidence_by_group": [{"group": g, "contribution": round(c, 4)} for g, c in groups],
    }


def group_attribution(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per evidence family, with each family's share of the total."""
    groups = [feature_group(n) for n in feature_names]
    frame = pd.DataFrame(shap_values, columns=feature_names)
    per_group = frame.T.groupby(groups).sum().T          # signed sum per row
    mean_abs = per_group.abs().mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    return (
        pd.DataFrame({
            "group": mean_abs.index,
            "mean_abs_shap": mean_abs.to_numpy(),
            "share": mean_abs.to_numpy() / total,
            "features": [groups.count(g) for g in mean_abs.index],
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

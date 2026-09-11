"""Feature layer (L2): preprocessing and feature engineering.

Stage 02 of the workflow figure. Produces, from the merged raw frame:

  base        calendar and amount transforms
  entity      the synthetic account id (uid) and its components
  velocity    per-card / per-device counts and sums over 1h, 24h, 168h windows
  recency     seconds since the entity's previous transaction
  rolling     expanding mean / std of amount per entity, and the z-score of the
              current amount against that history
  graph       degree and distinct-partner counts over the card-merchant-device
              graph, plus causal neighbourhood fraud rates
  mismatch    email, address and device consistency flags
  encoded     frequency and causal target encodings of the categorical fields

Every aggregate is causal (see ``frds.features.causal``); every encoding that
needs a fitted statistic is fitted on the training span only.

A second property matters as much as causality and is easier to overlook:
*stationarity*. A feature can be perfectly causal and still be useless at
deployment if its distribution moves simply because time has passed. A
cumulative transaction count is the clearest case - a device with 500 prior
transactions in month six is not the same kind of object as one with 500 in
month one, but the column cannot say so, and a tree that split on 500 during
training is splitting on something else entirely by the test period. The drift
report measured exactly this: the worst-drifting columns were the cumulative
counters and the raw day index.

``features.stationarise`` addresses it by dropping the bare time index,
re-expressing the D columns against a fixed origin, and converting cumulative
counters into rates and shares. It is a switch rather than a default so the
effect can be measured instead of assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.split import SplitIndex
from ..utils import get_logger, write_json
from . import causal

log = get_logger("frds.features.build")

SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600


@dataclass
class FeatureSpec:
    """What the feature stage produced, so training and serving agree."""

    columns: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_features": len(self.columns),
            "columns": self.columns,
            "categorical": self.categorical,
            "groups": {k: len(v) for k, v in self.groups.items()},
            "group_members": self.groups,
            "dropped": {k: len(v) for k, v in self.dropped.items()},
        }


# ---------------------------------------------------------------- base ----
def add_base_features(df: pd.DataFrame, cfg: Config, out: dict) -> list[str]:
    time_col = cfg.require("data.time_col")
    amount_col = cfg.require("data.amount_col")

    t = df[time_col].to_numpy(dtype=np.int64)
    amt = df[amount_col].to_numpy(dtype=np.float64)

    stationarise = bool(cfg.get("features.stationarise", False))
    # The raw day index is the single worst-drifting column in the feature set
    # (PSI above 11): it is a monotone function of time, so every test-span row
    # falls on one side of any split learned during training. Hour and day of
    # week are cyclical and stay.
    if not stationarise:
        out["txn_day"] = (t // SECONDS_PER_DAY).astype(np.int32)
    out["txn_hour"] = ((t // SECONDS_PER_HOUR) % 24).astype(np.int8)
    out["txn_dow"] = ((t // SECONDS_PER_DAY) % 7).astype(np.int8)
    out["amt_log"] = np.log1p(amt).astype(np.float32)
    # A round amount and a long decimal tail behave very differently; the
    # cents part is one of the stronger cheap signals on IEEE-CIS.
    out["amt_cents"] = np.round(amt - np.floor(amt), 3).astype(np.float32)
    out["amt_is_round"] = (out["amt_cents"] == 0).astype(np.int8)
    out["amt_decimal_len"] = (
        pd.Series(amt).astype(str).str.split(".").str[-1].str.len().to_numpy()
    ).astype(np.int8)
    created = ["txn_hour", "txn_dow", "amt_log", "amt_cents",
               "amt_is_round", "amt_decimal_len"]
    return (created if stationarise else ["txn_day"] + created)


# -------------------------------------------------------------- entity ----
def _as_text(series: pd.Series) -> pd.Series:
    """Column to plain strings, with missing values as an explicit level.

    ``astype(str)`` is not enough: under the pandas 3 string dtype a missing
    value survives the cast as NA rather than becoming the text "nan", and it
    then breaks any join that follows. "Missing" is a real level here anyway -
    a card with no billing address is a different entity from one that has one.
    """
    return series.astype("string").fillna("__na__").astype("object")


def _join_columns(parts: list[pd.Series]) -> pd.Series:
    """Concatenate string columns with a separator, vectorised.

    A row-wise ``.agg("|".join, axis=1)`` is a Python loop over every row; at
    590,540 rows that alone costs more than the rest of the feature stage.
    """
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "|" + part
    return joined


def build_entity_keys(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """Dense integer codes for the entities the graph and counters use.

    ``uid`` is the standard IEEE-CIS account proxy: a card is anchored to the
    day it was first seen (``txn_day - D1``), which separates two different
    accounts that happen to share a card1 value.
    """
    components = [c for c in cfg.require("features.uid_components") if c in df.columns]
    day = (df[cfg.require("data.time_col")].to_numpy(np.int64) // SECONDS_PER_DAY)

    keys: dict[str, np.ndarray] = {}
    keys["card"] = causal.encode_groups(df["card1"]) if "card1" in df else np.zeros(len(df), np.int64)

    parts = [_as_text(df[c]) for c in components]
    if "D1" in df.columns:
        origin = np.round(day - df["D1"].fillna(-1).to_numpy(np.float64), 0)
        parts.append(pd.Series(origin.astype(np.int64).astype(str), index=df.index))
    keys["uid"] = causal.encode_groups(_join_columns(parts))

    keys["device"] = causal.encode_groups(
        df["DeviceInfo"] if "DeviceInfo" in df else pd.Series(np.zeros(len(df)))
    )
    keys["address"] = causal.encode_groups(
        df["addr1"] if "addr1" in df else pd.Series(np.zeros(len(df)))
    )
    keys["email"] = causal.encode_groups(
        df["P_emaildomain"] if "P_emaildomain" in df else pd.Series(np.zeros(len(df)))
    )
    # "Merchant" is not a column in IEEE-CIS. ProductCD x billing region is the
    # closest available stand-in, and is what the graph uses as its merchant
    # node. Stated plainly here rather than left implicit.
    product = df["ProductCD"] if "ProductCD" in df else pd.Series("?", index=df.index)
    region = df["addr1"] if "addr1" in df else pd.Series(0, index=df.index)
    keys["merchant"] = causal.encode_groups(
        _join_columns([_as_text(product), _as_text(region)])
    )

    for name, codes in keys.items():
        log.info("entity %-9s distinct=%d", name, int(codes.max()) + 1)
    return keys


# ------------------------------------------------------------ velocity ----
def add_velocity_features(
    df: pd.DataFrame, cfg: Config, keys: dict[str, np.ndarray], out: dict
) -> list[str]:
    time_col = cfg.require("data.time_col")
    amount_col = cfg.require("data.amount_col")
    t = df[time_col].to_numpy(np.int64)
    amt = df[amount_col].to_numpy(np.float64)
    windows = cfg.require("features.velocity_windows_hours")
    stationarise = bool(cfg.get("features.stationarise", False))

    created: list[str] = []
    for entity in ("uid", "card", "device", "address"):
        codes = keys[entity]
        for hours in windows:
            secs = int(hours) * SECONDS_PER_HOUR
            cnt = f"vel_{entity}_cnt_{hours}h"
            amtsum = f"vel_{entity}_amt_{hours}h"
            out[cnt] = causal.count_in_window(codes, t, secs)
            out[amtsum] = causal.sum_in_window(codes, t, amt, secs)
            created += [cnt, amtsum]
        # Total history length is a different signal from a windowed burst -
        # but as a raw count it grows with calendar time and cannot transfer.
        # Expressed per day of entity age it is stationary and still carries
        # "how busy is this entity".
        total = f"vel_{entity}_cnt_total"
        counts = causal.cumulative_count(codes).astype(np.float32)
        if stationarise:
            age_days = causal.seconds_since_first_seen(codes, t) / SECONDS_PER_DAY
            out[f"vel_{entity}_rate_per_day"] = (
                counts / np.maximum(age_days, 1.0)
            ).astype(np.float32)
            out[f"vel_{entity}_age_days"] = age_days.astype(np.float32)
            created += [f"vel_{entity}_rate_per_day", f"vel_{entity}_age_days"]
        else:
            out[total] = counts
            created.append(total)
    return created


def add_recency_and_rolling(
    df: pd.DataFrame, cfg: Config, keys: dict[str, np.ndarray], out: dict
) -> list[str]:
    time_col = cfg.require("data.time_col")
    amount_col = cfg.require("data.amount_col")
    t = df[time_col].to_numpy(np.int64)
    amt = df[amount_col].to_numpy(np.float64)

    created: list[str] = []
    for entity in ("uid", "card", "device"):
        codes = keys[entity]
        gap = f"recency_{entity}_seconds"
        out[gap] = causal.seconds_since_previous(codes, t)
        created.append(gap)

    for entity in ("uid", "card"):
        codes = keys[entity]
        mean_col = f"roll_{entity}_amt_mean"
        std_col = f"roll_{entity}_amt_std"
        z_col = f"roll_{entity}_amt_z"
        ratio_col = f"roll_{entity}_amt_ratio"
        mu = causal.expanding_mean(codes, amt)
        sd = causal.expanding_std(codes, amt)
        out[mean_col] = mu
        out[std_col] = sd
        with np.errstate(invalid="ignore", divide="ignore"):
            out[z_col] = ((amt - mu) / np.where(sd > 1e-6, sd, np.nan)).astype(np.float32)
            out[ratio_col] = (amt / np.where(mu > 1e-6, mu, np.nan)).astype(np.float32)
        created += [mean_col, std_col, z_col, ratio_col]
    return created


# --------------------------------------------------------------- graph ----
def add_graph_features(
    df: pd.DataFrame, cfg: Config, keys: dict[str, np.ndarray], out: dict,
    target: np.ndarray,
) -> list[str]:
    """Degree and neighbourhood statistics over the card-merchant-device graph.

    This is the tabular projection of the relational signal (gap G2). The
    heterogeneous GNN consumes the same graph directly; these columns let the
    boosting models see a shadow of it, and give an honest read on how much of
    the graph's value a tabular model can already capture.
    """
    stationarise = bool(cfg.get("features.stationarise", False))
    times = df[cfg.require("data.time_col")].to_numpy(np.int64)

    created: list[str] = []
    edges = [
        ("device", "card"),
        ("card", "device"),
        ("merchant", "card"),
        ("card", "merchant"),
        ("address", "card"),
        ("email", "card"),
        ("device", "uid"),
    ]
    # Degree is computed once and reused: velocity's *_cnt_total and graph's
    # *_degree were previously the same quantity under two names, which wasted
    # four columns and made the drift table read as if two things were failing.
    degree = {
        e: causal.cumulative_count(keys[e]).astype(np.float32)
        for e in ("card", "device", "merchant", "address", "email", "uid")
    }

    for left, right in edges:
        col = f"graph_{left}_distinct_{right}"
        distinct = causal.distinct_partners_so_far(keys[left], keys[right]).astype(np.float32)
        if stationarise:
            # The share of an entity's interactions that were with a partner it
            # had not seen before. A device used by eleven cards out of twelve
            # transactions is anomalous at any point in the timeline; eleven
            # cards out of four hundred is not.
            out[f"{col}_share"] = (distinct / np.maximum(degree[left], 1.0)).astype(np.float32)
            created.append(f"{col}_share")
        else:
            out[col] = distinct
            created.append(col)

    for entity in ("card", "device", "merchant", "address", "email", "uid"):
        if stationarise:
            age_days = causal.seconds_since_first_seen(keys[entity], times) / SECONDS_PER_DAY
            col = f"graph_{entity}_degree_per_day"
            out[col] = (degree[entity] / np.maximum(age_days, 1.0)).astype(np.float32)
        else:
            col = f"graph_{entity}_degree"
            out[col] = degree[entity]
        created.append(col)

    # Causal neighbourhood fraud rate: how risky this entity's own history has
    # been. Shrunk towards the global prior, and never sees the current label.
    #
    # These are the features that carry the labelling assumption of section 4.3.
    # `features.label_delay_days` withholds outcomes that would not yet have
    # been confirmed, and `features.disable_target_encoding` drops them
    # entirely - together they let the ablation measure how much of the
    # headline result depends on knowing outcomes early.
    if cfg.get("features.disable_target_encoding", False):
        log.warning("target-encoding features disabled by config")
        return created

    delay_days = float(cfg.get("features.label_delay_days", 0))
    delay_seconds = int(delay_days * SECONDS_PER_DAY)
    if delay_seconds:
        log.info("label delay: outcomes withheld for %.0f days", delay_days)
    times = df[cfg.require("data.time_col")].to_numpy(np.int64)

    for entity in ("card", "device", "merchant", "uid", "email"):
        col = f"graph_{entity}_prior_fraud_rate"
        out[col] = causal.expanding_target_rate(
            keys[entity], target, times=times, label_delay_seconds=delay_seconds
        )
        created.append(col)

    # A card whose device pool is wide relative to its history is anomalous.
    # Under stationarise this is already covered by the *_share columns above.
    if not stationarise:
        with np.errstate(invalid="ignore", divide="ignore"):
            out["graph_card_device_spread"] = (
                out["graph_card_distinct_device"].astype(np.float32)
                / np.maximum(out["graph_card_degree"].astype(np.float32), 1.0)
            )
        created.append("graph_card_device_spread")
    return created



# --------------------------------------------------- entity aggregates ----
def add_entity_aggregates(
    df: pd.DataFrame, cfg: Config, keys: dict[str, np.ndarray], split: SplitIndex,
    out: dict,
) -> list[str]:
    """Per-entity norms, and this transaction's deviation from them.

    A transaction of 400 currency units means nothing on its own. On an account
    whose every previous transaction was near 20, it means a great deal. The
    ratio of a value to the entity's own running mean is what carries that, and
    it is the pattern that separates strong solutions on this benchmark from
    average ones.

    Everything here is behavioural rather than label-derived, so unlike the
    prior-fraud-rate features it does not depend on outcomes being confirmed
    quickly. That is the point: these are the features expected to survive a
    label-confirmation delay.
    """
    if not cfg.get("features.entity_aggregates", False):
        return []

    entities = list(cfg.get("features.aggregate_entities", ["uid", "card", "device"]))
    value_cols = [
        c for c in cfg.get("features.aggregate_value_columns", [])
        if c in df.columns
    ]
    if not value_cols:
        log.warning("entity aggregates enabled but no value columns present")
        return []

    created: list[str] = []
    for entity in entities:
        if entity not in keys:
            continue
        codes = keys[entity]
        for col in value_cols:
            values = df[col].to_numpy(np.float64)
            mean = causal.expanding_mean(codes, values)
            mean_col = f"agg_{entity}_{col}_mean"
            ratio_col = f"agg_{entity}_{col}_ratio"
            out[mean_col] = mean
            with np.errstate(invalid="ignore", divide="ignore"):
                out[ratio_col] = (
                    values / np.where(np.abs(mean) > 1e-6, mean, np.nan)
                ).astype(np.float32)
            created += [mean_col, ratio_col]

    # Alternative account definitions. The uid used elsewhere anchors a card to
    # the day it was first seen; these two do not, so they group differently
    # and disagree with it in useful ways. Frequency-encoded on the training
    # span only, like every other categorical.
    min_freq = int(cfg.require("features.min_category_frequency"))
    alternates = {
        "uid_card_addr": ["card1", "addr1"],
        "uid_card_addr_email": ["card1", "addr1", "P_emaildomain"],
    }
    for name, components in alternates.items():
        present = [c for c in components if c in df.columns]
        if len(present) < 2:
            continue
        series = _join_columns([_as_text(df[c]) for c in present])
        counts = series.iloc[split.train].value_counts()
        counts = counts[counts >= min_freq]
        col = f"freq_{name}"
        out[col] = series.map(counts).fillna(0).to_numpy(np.float32)
        created.append(col)

    log.info("entity aggregates: %d columns over %s", len(created), entities)
    return created


# ------------------------------------------------------------ mismatch ----
def add_mismatch_features(df: pd.DataFrame, out: dict) -> list[str]:
    created: list[str] = []
    if "P_emaildomain" in df and "R_emaildomain" in df:
        p = df["P_emaildomain"].astype("string")
        r = df["R_emaildomain"].astype("string")
        out["mismatch_email_domain"] = (
            (p.notna() & r.notna() & (p != r)).astype(np.int8).to_numpy()
        )
        out["mismatch_email_suffix"] = (
            (p.str.split(".").str[-1] != r.str.split(".").str[-1])
            .fillna(False).astype(np.int8).to_numpy()
        )
        out["missing_r_email"] = r.isna().astype(np.int8).to_numpy()
        created += ["mismatch_email_domain", "mismatch_email_suffix", "missing_r_email"]

    if "dist1" in df and "dist2" in df:
        out["mismatch_geo_distance"] = (
            df[["dist1", "dist2"]].max(axis=1).to_numpy(np.float32)
        )
        out["missing_geo"] = df["dist1"].isna().astype(np.int8).to_numpy()
        created += ["mismatch_geo_distance", "missing_geo"]

    if "DeviceInfo" in df:
        out["missing_device"] = df["DeviceInfo"].isna().astype(np.int8).to_numpy()
        created.append("missing_device")

    m_cols = [c for c in df.columns if c.startswith("M") and c[1:].isdigit()]
    if m_cols:
        as_bool = df[m_cols].apply(lambda s: s.map({"T": 1, "F": 0}))
        out["match_flag_true_count"] = as_bool.sum(axis=1).to_numpy(np.float32)
        out["match_flag_missing_count"] = df[m_cols].isna().sum(axis=1).to_numpy(np.float32)
        created += ["match_flag_true_count", "match_flag_missing_count"]

    out["null_count_row"] = df.isna().sum(axis=1).to_numpy(np.float32)
    created.append("null_count_row")
    return created


# ------------------------------------------------------------- encoded ----
CATEGORICAL_CANDIDATES = (
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
    "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
)


def add_encoded_categoricals(
    df: pd.DataFrame, cfg: Config, split: SplitIndex, out: dict
) -> tuple[list[str], list[str]]:
    """Frequency-encode every categorical, and label-encode it for LightGBM.

    Frequencies are counted on the training span only. A level that appears
    for the first time in the test span therefore gets a count of zero, which
    is both correct and informative.
    """
    min_freq = int(cfg.require("features.min_category_frequency"))
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    present = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    for col in present:
        series = df[col].astype("string").fillna("__missing__")
        train_counts = series.iloc[split.train].value_counts()
        kept = train_counts[train_counts >= min_freq].index
        collapsed = series.where(series.isin(kept), "__rare__")

        freq_col = f"freq_{col}"
        out[freq_col] = collapsed.map(train_counts).fillna(0).to_numpy(np.float32)
        numeric_cols.append(freq_col)

        cat_col = f"cat_{col}"
        # Categories are taken from the training span; unseen levels become -1.
        categories = pd.Index(sorted(set(collapsed.iloc[split.train].unique())))
        codes = pd.Categorical(collapsed, categories=categories).codes
        out[cat_col] = codes.astype(np.int32)
        categorical_cols.append(cat_col)

    log.info("encoded %d categorical columns", len(present))
    return numeric_cols, categorical_cols


# ------------------------------------------------------- column pruning ----
def select_raw_numeric(df: pd.DataFrame, cfg: Config, split: SplitIndex) -> tuple[list[str], dict]:
    """Keep the informative slice of the raw numeric columns.

    IEEE-CIS ships 339 V columns that are highly redundant. They are grouped by
    their missingness pattern - within a group the columns were computed on the
    same population - and inside each group all but the least-correlated
    representatives are dropped. Decisions are made on the training span only.
    """
    drop_null = float(cfg.require("features.drop_null_threshold"))
    max_v = int(cfg.require("features.max_v_columns"))
    exclude = {
        cfg.require("data.id_col"), cfg.require("data.time_col"),
        cfg.require("data.target_col"),
    }

    numeric = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    train = df.iloc[split.train]

    dropped_null = [c for c in numeric if train[c].isna().mean() > drop_null]
    dropped_const = [
        c for c in numeric
        if c not in dropped_null and train[c].nunique(dropna=True) <= 1
    ]
    keep = [c for c in numeric if c not in set(dropped_null) | set(dropped_const)]

    v_cols = [c for c in keep if c.startswith("V") and c[1:].isdigit()]
    dropped_v: list[str] = []
    if len(v_cols) > max_v:
        dropped_v = _prune_v_columns(train, v_cols, max_v)
        keep = [c for c in keep if c not in set(dropped_v)]

    report = {
        "high_null": dropped_null,
        "constant": dropped_const,
        "redundant_v": dropped_v,
    }
    log.info(
        "raw numeric: kept %d, dropped %d null / %d constant / %d redundant-V",
        len(keep), len(dropped_null), len(dropped_const), len(dropped_v),
    )
    return keep, report


def _prune_v_columns(train: pd.DataFrame, v_cols: list[str], max_keep: int) -> list[str]:
    """Group V columns by NaN pattern, keep the most distinctive per group."""
    null_signature = train[v_cols].isna().sum(axis=0)
    groups: dict[int, list[str]] = {}
    for col in v_cols:
        groups.setdefault(int(null_signature[col]), []).append(col)

    # Budget proportionally: bigger blocks keep more representatives.
    keep: list[str] = []
    for _, members in sorted(groups.items()):
        quota = max(1, round(max_keep * len(members) / len(v_cols)))
        if len(members) <= quota:
            keep.extend(members)
            continue
        sub = train[members]
        # Rank by distinct-value count: a column that varies more carries more.
        ranked = sub.nunique(dropna=True).sort_values(ascending=False).index.tolist()
        chosen: list[str] = []
        for cand in ranked:
            if len(chosen) >= quota:
                break
            if not chosen:
                chosen.append(cand)
                continue
            corr = sub[chosen + [cand]].corr().loc[cand, chosen].abs().max()
            if pd.isna(corr) or corr < 0.95:
                chosen.append(cand)
        keep.extend(chosen or ranked[:quota])

    return [c for c in v_cols if c not in set(keep)]


# ------------------------------------------------------------ assembly ----
def build_features(
    df: pd.DataFrame, cfg: Config, split: SplitIndex
) -> tuple[pd.DataFrame, pd.Series, FeatureSpec, dict[str, np.ndarray]]:
    target_col = cfg.require("data.target_col")
    y = df[target_col].astype(np.int8)
    target = y.to_numpy()

    out: dict[str, np.ndarray] = {}
    spec = FeatureSpec()

    spec.groups["base"] = add_base_features(df, cfg, out)
    keys = build_entity_keys(df, cfg)
    spec.groups["velocity"] = add_velocity_features(df, cfg, keys, out)
    spec.groups["recency_rolling"] = add_recency_and_rolling(df, cfg, keys, out)
    spec.groups["graph"] = add_graph_features(df, cfg, keys, out, target)
    spec.groups["entity_aggregates"] = add_entity_aggregates(df, cfg, keys, split, out)
    spec.groups["mismatch"] = add_mismatch_features(df, out)

    freq_cols, cat_cols = add_encoded_categoricals(df, cfg, split, out)
    spec.groups["encoded_frequency"] = freq_cols
    spec.groups["encoded_label"] = cat_cols
    spec.categorical = cat_cols

    raw_numeric, dropped = select_raw_numeric(df, cfg, split)
    stationarise = bool(cfg.get("features.stationarise", False))
    day = df[cfg.require("data.time_col")].to_numpy(np.int64) // SECONDS_PER_DAY
    for col in raw_numeric:
        values = df[col].to_numpy(np.float32)
        if stationarise and col.startswith("D") and col[1:].isdigit():
            # D columns count days since some earlier event, so they grow with
            # calendar time. Subtracting the day index converts each into the
            # (fixed) day that event happened on, which does not drift.
            values = (day - values).astype(np.float32)
        out[col] = values
    spec.groups["raw_numeric"] = raw_numeric
    spec.dropped = dropped

    # Assembled in one shot: inserting 250 columns one at a time fragments the
    # frame badly enough for pandas to warn about it.
    frame = pd.DataFrame(out, index=df.index)
    # Infinities arise from the ratio features when a denominator is tiny.
    frame = frame.replace([np.inf, -np.inf], np.nan)
    out = frame
    spec.columns = list(out.columns)

    log.info(
        "feature matrix %s | %s",
        out.shape,
        ", ".join(f"{k}={len(v)}" for k, v in spec.groups.items()),
    )
    return out, y, spec, keys


def save_features(
    X: pd.DataFrame, y: pd.Series, spec: FeatureSpec, keys: dict[str, np.ndarray], cfg: Config
) -> Path:
    processed = cfg.path("processed")
    processed.mkdir(parents=True, exist_ok=True)
    frame = X.copy()
    frame[cfg.require("data.target_col")] = y.to_numpy()
    for name, codes in keys.items():
        frame[f"__entity_{name}"] = codes
    path = processed / "features.parquet"
    frame.to_parquet(path, index=False)
    write_json(spec.to_dict(), processed / "feature_spec.json")
    log.info("wrote %s", path.name)
    return path


def load_features(cfg: Config) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, np.ndarray]]:
    path = cfg.path("processed") / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m frds features' first.")
    frame = pd.read_parquet(path)
    target_col = cfg.require("data.target_col")
    y = frame.pop(target_col)
    keys = {}
    for col in [c for c in frame.columns if c.startswith("__entity_")]:
        keys[col.removeprefix("__entity_")] = frame.pop(col).to_numpy()
    from ..utils import read_json

    spec = read_json(cfg.path("processed") / "feature_spec.json")
    return frame, y, spec["categorical"], keys

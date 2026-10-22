"""Feature layer (L2): preprocessing and feature engineering."""

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

# Which entities each feature family is computed over. Shared with the online
# scorer (features/online.py), so the two cannot disagree about the feature set.
VELOCITY_ENTITIES = ("uid", "card", "device", "address")
RECENCY_ENTITIES = ("uid", "card", "device")
ROLLING_ENTITIES = ("uid", "card")
GRAPH_EDGES = (
    ("device", "card"),
    ("card", "device"),
    ("merchant", "card"),
    ("card", "merchant"),
    ("address", "card"),
    ("email", "card"),
    ("device", "uid"),
)
#: Extra relations profiled when ``features.account_profile`` is on.
PROFILE_EDGES = (
    ("uid", "device"),
    ("uid", "merchant"),
    ("card", "email"),
    ("card", "address"),
)
DEGREE_ENTITIES = ("card", "device", "merchant", "address", "email", "uid")
PRIOR_RATE_ENTITIES = ("card", "device", "merchant", "uid", "email")
PRIOR_WEIGHT = 20.0


@dataclass
class FeatureSpec:
    """What the feature stage produced, so training and serving agree."""

    columns: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)
    dropped: dict[str, list[str]] = field(default_factory=dict)
    #: statistics fitted on the training span (encodings, priors) that the
    #: online scorer must reuse verbatim; saved alongside, not in the JSON
    fitted: dict = field(default_factory=dict)

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
    # The raw day index drifts badly (monotone in time); hour and weekday are cyclical and kept.
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


def graph_edges(cfg: Config) -> tuple[tuple[str, str], ...]:
    """The relations profiled for distinct-partner counts."""
    if cfg.get("features.account_profile", False):
        return GRAPH_EDGES + PROFILE_EDGES
    return GRAPH_EDGES


def aggregate_value_columns(cfg: Config, available) -> list[str]:
    """Columns averaged per entity: the base set, plus the account profile."""
    columns = list(cfg.get("features.aggregate_value_columns", []) or [])
    if cfg.get("features.account_profile", False):
        columns += [c for c in cfg.get("features.profile_value_columns", []) or []
                    if c not in columns]
    present = set(available)
    return [c for c in columns if c in present]


# -------------------------------------------------------------- entity ----
def _as_text(series: pd.Series) -> pd.Series:
    """Column to plain strings, with missing values as an explicit level."""
    return series.astype("string").fillna("__na__").astype("object")


def _join_columns(parts: list[pd.Series]) -> pd.Series:
    """Concatenate string columns with a separator, vectorised."""
    joined = parts[0]
    for part in parts[1:]:
        joined = joined + "|" + part
    return joined


def entity_key_text(df: pd.DataFrame, cfg: Config) -> dict[str, pd.Series]:
    """The identity of every entity a transaction touches, as plain text."""
    def column(name: str) -> pd.Series:
        if name in df.columns:
            return _as_text(df[name])
        return pd.Series("__na__", index=df.index, dtype="object")

    components = [c for c in cfg.require("features.uid_components") if c in df.columns]
    day = (df[cfg.require("data.time_col")].to_numpy(np.int64) // SECONDS_PER_DAY)

    parts = [_as_text(df[c]) for c in components]
    if "D1" in df.columns:
        origin = np.round(day - df["D1"].fillna(-1).to_numpy(np.float64), 0)
        parts.append(pd.Series(origin.astype(np.int64).astype(str), index=df.index))

    return {
        "card": column("card1"),
        "uid": _join_columns(parts) if parts else column("card1"),
        "device": column("DeviceInfo"),
        "address": column("addr1"),
        "email": column("P_emaildomain"),
        # IEEE-CIS has no merchant column; ProductCD x billing region stands in for it.
        "merchant": _join_columns([column("ProductCD"), column("addr1")]),
    }


def build_entity_keys(df: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """Dense integer codes for the entities the graph and counters use."""
    keys = {name: causal.encode_groups(text)
            for name, text in entity_key_text(df, cfg).items()}
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
    for entity in VELOCITY_ENTITIES:
        codes = keys[entity]
        for hours in windows:
            secs = int(hours) * SECONDS_PER_HOUR
            cnt = f"vel_{entity}_cnt_{hours}h"
            amtsum = f"vel_{entity}_amt_{hours}h"
            out[cnt] = causal.count_in_window(codes, t, secs)
            out[amtsum] = causal.sum_in_window(codes, t, amt, secs)
            created += [cnt, amtsum]
        # Total history per day of entity age: stationary, unlike a raw count.
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
    for entity in RECENCY_ENTITIES:
        codes = keys[entity]
        gap = f"recency_{entity}_seconds"
        out[gap] = causal.seconds_since_previous(codes, t)
        created.append(gap)

    for entity in ROLLING_ENTITIES:
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
    target: np.ndarray, global_rate: float | None = None,
) -> list[str]:
    """Degree and neighbourhood statistics over the card-merchant-device graph."""
    stationarise = bool(cfg.get("features.stationarise", False))
    times = df[cfg.require("data.time_col")].to_numpy(np.int64)

    created: list[str] = []
    edges = graph_edges(cfg)
    # Degree is computed once and shared by the velocity and graph features.
    degree = {
        e: causal.cumulative_count(keys[e]).astype(np.float32)
        for e in DEGREE_ENTITIES
    }

    for left, right in edges:
        col = f"graph_{left}_distinct_{right}"
        distinct = causal.distinct_partners_so_far(keys[left], keys[right]).astype(np.float32)
        if stationarise:
            # Share of interactions with a previously unseen partner.
            out[f"{col}_share"] = (distinct / np.maximum(degree[left], 1.0)).astype(np.float32)
            created.append(f"{col}_share")
        else:
            out[col] = distinct
            created.append(col)

    for entity in DEGREE_ENTITIES:
        if stationarise:
            age_days = causal.seconds_since_first_seen(keys[entity], times) / SECONDS_PER_DAY
            col = f"graph_{entity}_degree_per_day"
            out[col] = (degree[entity] / np.maximum(age_days, 1.0)).astype(np.float32)
        else:
            col = f"graph_{entity}_degree"
            out[col] = degree[entity]
        created.append(col)

    # Causal prior fraud rate per entity, shrunk to the global prior; never sees the current label.
    # label_delay_days and disable_target_encoding drive the label-delay ablation.
    if cfg.get("features.disable_target_encoding", False):
        log.warning("target-encoding features disabled by config")
        return created

    delay_days = float(cfg.get("features.label_delay_days", 0))
    delay_seconds = int(delay_days * SECONDS_PER_DAY)
    if delay_seconds:
        log.info("label delay: outcomes withheld for %.0f days", delay_days)
    times = df[cfg.require("data.time_col")].to_numpy(np.int64)

    for entity in PRIOR_RATE_ENTITIES:
        col = f"graph_{entity}_prior_fraud_rate"
        out[col] = causal.expanding_target_rate(
            keys[entity], target, times=times, label_delay_seconds=delay_seconds,
            global_rate=global_rate, prior_weight=PRIOR_WEIGHT,
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
    out: dict, fitted: dict | None = None,
) -> list[str]:
    """Per-entity norms, and this transaction's deviation from them."""
    if not cfg.get("features.entity_aggregates", False):
        return []

    entities = list(cfg.get("features.aggregate_entities", ["uid", "card", "device"]))
    value_cols = aggregate_value_columns(cfg, df.columns)
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

    # Alternative account keys, frequency-encoded on the training span.
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
        if fitted is not None:
            fitted.setdefault("alternate_ids", {})[col] = {
                "components": present,
                "counts": {str(k): float(v) for k, v in counts.items()},
            }

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
    df: pd.DataFrame, cfg: Config, split: SplitIndex, out: dict,
    fitted: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Frequency-encode every categorical, and label-encode it for LightGBM."""
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

        if fitted is not None:
            # Saved for the online scorer: surviving levels, their counts and the label order.
            fitted.setdefault("categoricals", {})[col] = {
                "counts": {str(k): float(v) for k, v in train_counts[kept].items()},
                "categories": [str(c) for c in categories],
            }

    log.info("encoded %d categorical columns", len(present))
    return numeric_cols, categorical_cols


# ------------------------------------------------------- column pruning ----
def select_raw_numeric(df: pd.DataFrame, cfg: Config, split: SplitIndex) -> tuple[list[str], dict]:
    """Keep the informative slice of the raw numeric columns."""
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
    # Shrinkage prior for the causal fraud-rate features, from the training
    # span only. Recorded so the online scorer shrinks towards the same value.
    global_rate = float(target[split.train].mean())
    spec.fitted["global_fraud_rate"] = global_rate

    spec.groups["base"] = add_base_features(df, cfg, out)
    keys = build_entity_keys(df, cfg)
    spec.groups["velocity"] = add_velocity_features(df, cfg, keys, out)
    spec.groups["recency_rolling"] = add_recency_and_rolling(df, cfg, keys, out)
    spec.groups["graph"] = add_graph_features(df, cfg, keys, out, target, global_rate)
    spec.groups["entity_aggregates"] = add_entity_aggregates(
        df, cfg, keys, split, out, spec.fitted
    )
    spec.groups["mismatch"] = add_mismatch_features(df, out)

    freq_cols, cat_cols = add_encoded_categoricals(df, cfg, split, out, spec.fitted)
    spec.groups["encoded_frequency"] = freq_cols
    spec.groups["encoded_label"] = cat_cols
    spec.categorical = cat_cols

    raw_numeric, dropped = select_raw_numeric(df, cfg, split)
    stationarise = bool(cfg.get("features.stationarise", False))
    day = df[cfg.require("data.time_col")].to_numpy(np.int64) // SECONDS_PER_DAY
    for col in raw_numeric:
        values = df[col].to_numpy(np.float32)
        if stationarise and col.startswith("D") and col[1:].isdigit():
            # D columns grow with time; day - D gives the fixed event day instead.
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

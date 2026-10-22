"""Feature layer (L2), online half: features for a transaction nobody has seen."""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger
from . import build, causal

log = get_logger("frds.features.online")

STATE_FILE = "online_state.joblib"
STATE_VERSION = 2

#: Feature-stage settings the stored history was built under. Serving with
#: different values would silently compute a different feature set.
_GUARDED_KEYS = (
    "features.stationarise", "features.velocity_windows_hours",
    "features.label_delay_days", "features.disable_target_encoding",
    "features.entity_aggregates", "features.aggregate_entities",
    "features.aggregate_value_columns", "features.uid_components",
    "features.account_profile", "features.profile_value_columns",
)


# ------------------------------------------------------------- persistence --
def save_state(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    spec: "build.FeatureSpec",
    cfg: Config,
) -> Path:
    """Persist everything the online store needs, next to features.parquet."""
    time_col = cfg.require("data.time_col")
    amount_col = cfg.require("data.amount_col")

    texts = build.entity_key_text(df, cfg)
    codes, index = {}, {}
    for name, text in texts.items():
        c, uniques = pd.factorize(text, use_na_sentinel=False)
        codes[name] = c.astype(np.int64)
        index[name] = {str(u): i for i, u in enumerate(uniques)}

    value_cols = build.aggregate_value_columns(cfg, df.columns)
    state = {
        "templates": _templates(df, y, cfg),
        "version": STATE_VERSION,
        "guarded": {k: cfg.get(k) for k in _GUARDED_KEYS},
        "columns": list(X.columns),
        "dtypes": {c: str(t) for c, t in X.dtypes.items()},
        "raw_columns": list(df.columns),
        "raw_dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "raw_numeric": list(spec.groups.get("raw_numeric", [])),
        "fitted": spec.fitted,
        "key_index": index,
        "history": {
            "id": df[cfg.require("data.id_col")].to_numpy(np.int64),
            "time": df[time_col].to_numpy(np.int64),
            "amount": df[amount_col].to_numpy(np.float64),
            "target": y.to_numpy().astype(np.int8),
            "codes": codes,
            "values": {c: df[c].to_numpy(np.float64) for c in value_cols},
        },
    }
    path = cfg.path("processed") / STATE_FILE
    joblib.dump(state, path, compress=3)
    log.info("wrote %s (%d rows of entity history)", path.name, len(df))
    return path


def _templates(df: pd.DataFrame, y: pd.Series, cfg: Config, per_class: int = 150) -> list[dict]:
    """Real transactions from the end of the timeline, as starting points."""
    drop = {cfg.require("data.id_col"), cfg.require("data.time_col"),
            cfg.require("data.target_col")}
    tail = df.iloc[int(len(df) * 0.85):]
    labels = y.to_numpy()[int(len(df) * 0.85):]
    rng = np.random.default_rng(int(cfg.get("project.seed", 42)))
    picked = []
    for label in (1, 0):
        idx = np.flatnonzero(labels == label)
        if idx.size:
            picked += [(int(i), label) for i in rng.choice(idx, min(per_class, idx.size), replace=False)]
    out = []
    for i, label in sorted(picked):
        row = tail.iloc[i]
        fields = {c: (None if (not isinstance(v, str) and pd.isna(v)) else
                      (v.item() if hasattr(v, "item") else v))
                  for c, v in row.items() if c not in drop}
        out.append({"was_fraud": bool(label), "fields": fields})
    return out


def load_state(cfg: Config) -> dict:
    path = cfg.path("processed") / STATE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Re-run 'python -m frds features' to build it."
        )
    state = joblib.load(path)
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(f"{path.name} is from an older build; re-run 'frds features'")
    drift = {k: (v, cfg.get(k)) for k, v in state["guarded"].items() if cfg.get(k) != v}
    if drift:
        raise RuntimeError(
            "feature settings changed since the online state was built "
            f"(stored vs current): {drift}. Re-run 'python -m frds features'."
        )
    return state


# ------------------------------------------------------------- the store ----
@dataclass
class _Entity:
    """One entity type's stored history, sorted by (code, time), with prefix sums."""

    t: np.ndarray                 # times, grouped by code, time-ordered within
    bounds: np.ndarray            # rows of code c are [bounds[c], bounds[c+1])
    csum_amt: np.ndarray
    csum_amt_sq: np.ndarray
    csum_y: np.ndarray
    values: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)

    def span(self, code: int) -> tuple[int, int]:
        if 0 <= code < len(self.bounds) - 1:
            return int(self.bounds[code]), int(self.bounds[code + 1])
        return 0, 0


@dataclass
class _Live:
    """Running totals for one entity's arrivals since the stored history."""

    times: list[int] = field(default_factory=list)       # non-decreasing
    cum_amt: list[float] = field(default_factory=lambda: [0.0])
    s2: float = 0.0
    values: dict[str, list[float]] = field(default_factory=dict)   # col -> [n, s1, s2]
    label_n: int = 0
    label_sum: float = 0.0
    labelled_t: list[int] = field(default_factory=list)  # only kept under a label delay
    labelled_cum: list[float] = field(default_factory=lambda: [0.0])
    partners: dict[str, set[int]] = field(default_factory=dict)  # new partners only


@dataclass
class _Arrival:
    """A transaction that reached the store after the stored history."""

    txn_id: int
    t: int
    codes: dict[str, int]
    label: int | None = None


@dataclass
class _Row:
    """A parsed raw transaction: typed values plus the small frame that the
    offline row-level functions are run on."""

    frame: pd.DataFrame
    values: dict[str, Any]
    txn_id: int
    t: int
    amount: float
    missing: int


def _prefix(x: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])


#: Raw fields the shared row-level functions (base, mismatch, entity keys) read.
_ROW_FIELDS = ("P_emaildomain", "R_emaildomain", "dist1", "dist2", "DeviceInfo",
               "card1", "addr1", "ProductCD", "D1") + tuple(f"M{i}" for i in range(1, 10))


class OnlineFeatureStore:
    """Computes the offline feature vector for one new transaction at a time."""

    def __init__(self, state: dict, cfg: Config, upto: int | None = None) -> None:
        self.cfg = cfg
        self.state = state
        self.columns: list[str] = state["columns"]
        self.dtypes: dict[str, str] = state["dtypes"]
        self.column_index = {c: i for i, c in enumerate(self.columns)}
        self.raw_columns: list[str] = state["raw_columns"]
        self.raw_dtypes: dict[str, str] = state["raw_dtypes"]
        self.fitted: dict = state["fitted"]

        self.time_col = cfg.require("data.time_col")
        self.amount_col = cfg.require("data.amount_col")
        self.id_col = cfg.require("data.id_col")
        self.target_col = cfg.require("data.target_col")
        self.stationarise = bool(cfg.get("features.stationarise", False))
        self.windows = [int(h) for h in cfg.require("features.velocity_windows_hours")]
        self.label_delay = int(float(cfg.get("features.label_delay_days", 0)) * build.SECONDS_PER_DAY)
        self.use_target = not bool(cfg.get("features.disable_target_encoding", False))
        self.agg_on = bool(cfg.get("features.entity_aggregates", False))
        self.agg_entities = [e for e in cfg.get("features.aggregate_entities", []) or []]
        self.global_rate = float(self.fitted["global_fraud_rate"])
        self.edges = build.graph_edges(cfg)

        uid_parts = [c for c in cfg.require("features.uid_components") if c in self.raw_columns]
        wanted = {self.time_col, self.amount_col, self.id_col, *uid_parts, *_ROW_FIELDS}
        self.row_fields = [c for c in self.raw_columns if c in wanted]
        self.raw_kind = {c: ("int" if t.startswith("int") else
                             "float32" if t == "float32" else
                             "float" if t.startswith("float") else "text")
                         for c, t in self.raw_dtypes.items()}

        hist = state["history"]
        n = len(hist["time"]) if upto is None else int(upto)
        self.history_rows = n
        times = hist["time"][:n]
        amount = hist["amount"][:n]
        target = hist["target"][:n].astype(np.float64)
        self.value_cols = list(hist["values"])
        values = {c: v[:n] for c, v in hist["values"].items()}
        self.latest_time = int(times.max()) if n else 0
        self.max_id = int(hist["id"][:n].max()) if n else 0

        self.key_index: dict[str, dict[str, int]] = state["key_index"]
        self.code_count = {e: len(ix) for e, ix in self.key_index.items()}
        self.extra_index: dict[str, dict[str, int]] = {e: {} for e in self.key_index}

        self.entities: dict[str, _Entity] = {}
        codes_by_entity = {e: c[:n] for e, c in hist["codes"].items()}
        for name, codes in codes_by_entity.items():
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            bounds = np.searchsorted(sorted_codes, np.arange(self.code_count[name] + 1))
            entity = _Entity(
                t=times[order],
                bounds=bounds,
                csum_amt=_prefix(amount[order]),
                csum_amt_sq=_prefix(amount[order] ** 2),
                csum_y=_prefix(target[order]),
            )
            if self.agg_on and name in self.agg_entities:
                for col, v in values.items():
                    vs = v[order]
                    observed = ~np.isnan(vs)
                    v0 = np.where(observed, vs, 0.0)
                    entity.values[col] = (_prefix(observed.astype(np.float64)),
                                          _prefix(v0), _prefix(v0 * v0))
            self.entities[name] = entity

        # Distinct partners per entity, plus the seen pairs to detect new ones.
        self.distinct_total: dict[tuple[str, str], np.ndarray] = {}
        self.pair_keys: dict[tuple[str, str], np.ndarray] = {}
        for left, right in self.edges:
            width = self.code_count[right] + 1_000_000
            keys = np.unique(codes_by_entity[left] * width + codes_by_entity[right])
            self.pair_keys[(left, right)] = keys
            self.distinct_total[(left, right)] = np.bincount(
                keys // width, minlength=self.code_count[left]
            )
        self._pair_width = {edge: self.code_count[edge[1]] + 1_000_000
                            for edge in self.edges}

        self.arrivals: list[_Arrival] = []
        self.arrival_position: dict[int, int] = {}
        self.live: dict[str, dict[int, _Live]] = {e: {} for e in self.key_index}
        self._categories = {
            col: {c: i for i, c in enumerate(spec["categories"])}
            for col, spec in self.fitted.get("categoricals", {}).items()
        }
        log.info("online store ready: %d history rows, %s", n,
                 ", ".join(f"{e}={c}" for e, c in self.code_count.items()))

    @classmethod
    def from_disk(cls, cfg: Config, upto: int | None = None) -> "OnlineFeatureStore":
        return cls(load_state(cfg), cfg, upto)

    # -- input --------------------------------------------------------------
    def frame(self, raw: Mapping[str, Any] | pd.Series) -> _Row:
        """Parse one raw transaction against the benchmark schema."""
        raw = dict(raw.items()) if isinstance(raw, pd.Series) else dict(raw)
        unknown = set(raw) - set(self.raw_columns)
        if unknown:
            raise ValueError(f"unknown transaction fields: {sorted(unknown)[:8]}")

        def absent(v) -> bool:
            return v is None or (not isinstance(v, str) and bool(pd.isna(v)))

        if absent(raw.get(self.time_col)):
            raw[self.time_col] = self.latest_time + 1
        if absent(raw.get(self.id_col)):
            raw[self.id_col] = self.max_id + 1
        # The label is never an input. It is filled with a constant so that the
        # row-level missing-value count matches a labelled training row.
        raw[self.target_col] = 0

        values: dict[str, Any] = {}
        missing = 0
        for col in self.raw_columns:
            v = raw.get(col)
            kind = self.raw_kind[col]
            if absent(v):
                if kind == "int":
                    raise ValueError(f"'{col}' is required")
                missing += 1
                values[col] = np.float32("nan") if kind == "float32" else (
                    np.nan if kind == "float" else None)
            elif kind == "int":
                values[col] = int(v)
            elif kind == "float32":
                values[col] = np.float32(v)
            elif kind == "float":
                values[col] = float(v)
            else:
                values[col] = str(v)

        small = pd.DataFrame({
            c: pd.Series([values[c]], dtype=self.raw_dtypes[c]) for c in self.row_fields
        })
        return _Row(
            frame=small,
            values=values,
            txn_id=int(values[self.id_col]),
            t=int(values[self.time_col]),
            amount=float(np.float64(values[self.amount_col])),
            missing=missing,
        )

    def _codes(self, row: _Row) -> dict[str, int]:
        codes = {}
        for name, text in build.entity_key_text(row.frame, self.cfg).items():
            key = str(text.iloc[0])
            code = self.key_index[name].get(key)
            if code is None:
                extra = self.extra_index[name]
                if key not in extra:
                    extra[key] = self.code_count[name] + len(extra)
                code = extra[key]
            codes[name] = code
        return codes

    # -- the feature vector -------------------------------------------------
    def features(self, raw: Mapping[str, Any] | pd.Series) -> pd.DataFrame:
        """The one-row feature frame, with the training frame's exact dtypes."""
        row = self.frame(raw)
        return self._features(row, self._codes(row)).astype(self.dtypes)

    def _features(self, row: _Row, codes: dict[str, int]) -> pd.DataFrame:
        """The feature vector as one float64 row in training column order."""
        t = row.t
        if t < self.latest_time:
            raise ValueError(
                f"transaction time {t} precedes the latest one seen ({self.latest_time}); "
                "the store only scores forward in time"
            )
        amt = np.array([row.amount], np.float64)
        out: dict[str, np.ndarray] = {}

        # Row-local groups: the offline functions themselves, on one row. The
        # missing-value count needs every field, so it comes from the parser.
        build.add_base_features(row.frame, self.cfg, out)
        build.add_mismatch_features(row.frame, out)
        out["null_count_row"] = np.array([row.missing], np.float32)

        self._velocity(t, codes, out)
        self._recency_and_rolling(t, amt, codes, out)
        self._graph(t, codes, out)
        self._aggregates(row, codes, out)
        self._encoded(row, out)
        self._raw_numeric(row, t, out)

        vector = np.full(len(self.columns), np.nan)
        for name, value in out.items():
            j = self.column_index.get(name)
            if j is not None:
                vector[j] = float(np.asarray(value).reshape(-1)[0])
        vector[np.isinf(vector)] = np.nan
        return pd.DataFrame(vector[None, :], columns=self.columns)

    # -- history access -----------------------------------------------------
    def _live(self, entity: str, code: int) -> _Live | None:
        return self.live[entity].get(code)

    def _count_sum_since(self, entity: str, code: int, lower: int) -> tuple[int, float]:
        e = self.entities[entity]
        lo, hi = e.span(code)
        start = lo + int(np.searchsorted(e.t[lo:hi], lower, side="left"))
        count = hi - start
        total = float(e.csum_amt[hi] - e.csum_amt[start])
        live = self._live(entity, code)
        if live is not None and live.times:
            k = bisect.bisect_left(live.times, lower)
            count += len(live.times) - k
            total += live.cum_amt[-1] - live.cum_amt[k]
        return count, total

    def _amount_moments(self, entity: str, code: int) -> tuple[int, float, float]:
        e = self.entities[entity]
        lo, hi = e.span(code)
        n = hi - lo
        s1 = float(e.csum_amt[hi] - e.csum_amt[lo])
        s2 = float(e.csum_amt_sq[hi] - e.csum_amt_sq[lo])
        live = self._live(entity, code)
        if live is not None:
            n += len(live.times)
            s1 += live.cum_amt[-1]
            s2 += live.s2
        return n, s1, s2

    def _value_moments(self, entity: str, code: int, col: str) -> tuple[float, float, float]:
        cnt, c1, c2 = self.entities[entity].values[col]
        lo, hi = self.entities[entity].span(code)
        n = float(cnt[hi] - cnt[lo])
        s1 = float(c1[hi] - c1[lo])
        s2 = float(c2[hi] - c2[lo])
        live = self._live(entity, code)
        if live is not None and col in live.values:
            ln, l1, l2 = live.values[col]
            n, s1, s2 = n + ln, s1 + l1, s2 + l2
        return n, s1, s2

    def _degree(self, entity: str, code: int) -> int:
        lo, hi = self.entities[entity].span(code)
        live = self._live(entity, code)
        return (hi - lo) + (len(live.times) if live is not None else 0)

    def _first_and_last(self, entity: str, code: int) -> tuple[int | None, int | None]:
        e = self.entities[entity]
        lo, hi = e.span(code)
        first = int(e.t[lo]) if hi > lo else None
        last = int(e.t[hi - 1]) if hi > lo else None
        live = self._live(entity, code)
        if live is not None and live.times:
            first = live.times[0] if first is None else first
            last = live.times[-1]
        return first, last

    # -- feature groups -----------------------------------------------------
    def _velocity(self, t, codes, out) -> None:
        for entity in build.VELOCITY_ENTITIES:
            code = codes[entity]
            for hours in self.windows:
                lower = max(t - hours * build.SECONDS_PER_HOUR, 0)
                count, total = self._count_sum_since(entity, code, lower)
                out[f"vel_{entity}_cnt_{hours}h"] = np.array([count], np.int32)
                out[f"vel_{entity}_amt_{hours}h"] = np.array([total], np.float32)
            degree = np.array([self._degree(entity, code)], np.float32)
            if self.stationarise:
                age = self._age_days(t, entity, code)
                out[f"vel_{entity}_rate_per_day"] = (degree / np.maximum(age, 1.0)).astype(np.float32)
                out[f"vel_{entity}_age_days"] = age.astype(np.float32)
            else:
                out[f"vel_{entity}_cnt_total"] = degree

    def _age_days(self, t: int, entity: str, code: int) -> np.ndarray:
        first, _ = self._first_and_last(entity, code)
        seconds = np.array([0.0 if first is None else t - first], np.float32)
        return seconds / build.SECONDS_PER_DAY

    def _recency_and_rolling(self, t, amt, codes, out) -> None:
        for entity in build.RECENCY_ENTITIES:
            _, last = self._first_and_last(entity, codes[entity])
            gap = np.nan if last is None else float(t - last)
            out[f"recency_{entity}_seconds"] = np.array([gap], np.float32)

        for entity in build.ROLLING_ENTITIES:
            n, s1, s2 = self._amount_moments(entity, codes[entity])
            count = np.array([n], np.float64)
            mu = causal.moments_to_mean(count, np.array([s1])).astype(np.float32)
            sd = causal.moments_to_std(count, np.array([s1]), np.array([s2])).astype(np.float32)
            out[f"roll_{entity}_amt_mean"] = mu
            out[f"roll_{entity}_amt_std"] = sd
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"roll_{entity}_amt_z"] = (
                    (amt - mu) / np.where(sd > 1e-6, sd, np.nan)).astype(np.float32)
                out[f"roll_{entity}_amt_ratio"] = (
                    amt / np.where(mu > 1e-6, mu, np.nan)).astype(np.float32)

    def _in_history(self, left: str, right: str, code: int, partner: int) -> bool:
        keys = self.pair_keys[(left, right)]
        key = code * self._pair_width[(left, right)] + partner
        pos = int(np.searchsorted(keys, key))
        return pos < len(keys) and keys[pos] == key

    def _distinct(self, left: str, right: str, codes: dict[str, int]) -> int:
        code = codes[left]
        totals = self.distinct_total[(left, right)]
        distinct = int(totals[code]) if code < len(totals) else 0
        live = self._live(left, code)
        if live is not None:
            distinct += len(live.partners.get(right, ()))
        return distinct

    def _graph(self, t, codes, out) -> None:
        degree = {e: np.array([self._degree(e, codes[e])], np.float32)
                  for e in build.DEGREE_ENTITIES}
        for left, right in self.edges:
            col = f"graph_{left}_distinct_{right}"
            distinct = np.array([self._distinct(left, right, codes)], np.float32)
            if self.stationarise:
                out[f"{col}_share"] = (distinct / np.maximum(degree[left], 1.0)).astype(np.float32)
            else:
                out[col] = distinct

        for entity in build.DEGREE_ENTITIES:
            if self.stationarise:
                age = self._age_days(t, entity, codes[entity])
                out[f"graph_{entity}_degree_per_day"] = (
                    degree[entity] / np.maximum(age, 1.0)).astype(np.float32)
            else:
                out[f"graph_{entity}_degree"] = degree[entity]

        if not self.use_target:
            return
        cutoff = max(t - self.label_delay, 0) if self.label_delay else None
        for entity in build.PRIOR_RATE_ENTITIES:
            code = codes[entity]
            e = self.entities[entity]
            lo, hi = e.span(code)
            known_hi = hi
            if cutoff is not None:
                known_hi = lo + int(np.searchsorted(e.t[lo:hi], cutoff, side="right"))
            count = known_hi - lo
            fraud = float(e.csum_y[known_hi] - e.csum_y[lo])
            live = self._live(entity, code)
            if live is not None:
                if cutoff is None:
                    count += live.label_n
                    fraud += live.label_sum
                else:
                    k = bisect.bisect_right(live.labelled_t, cutoff)
                    count += k
                    fraud += live.labelled_cum[k]
            rate = (fraud + build.PRIOR_WEIGHT * self.global_rate) / (count + build.PRIOR_WEIGHT)
            out[f"graph_{entity}_prior_fraud_rate"] = np.array([rate], np.float32)

        if not self.stationarise:
            with np.errstate(invalid="ignore", divide="ignore"):
                out["graph_card_device_spread"] = (
                    out["graph_card_distinct_device"].astype(np.float32)
                    / np.maximum(out["graph_card_degree"].astype(np.float32), 1.0))

    def _aggregates(self, row: _Row, codes, out) -> None:
        if not self.agg_on:
            return
        for entity in self.agg_entities:
            if entity not in codes:
                continue
            for col in self.value_cols:
                values = np.array([row.values[col]], np.float32).astype(np.float64)
                n, s1, _ = self._value_moments(entity, codes[entity], col)
                mean = causal.moments_to_mean(np.array([n]), np.array([s1])).astype(np.float32)
                out[f"agg_{entity}_{col}_mean"] = mean
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[f"agg_{entity}_{col}_ratio"] = (
                        values / np.where(np.abs(mean) > 1e-6, mean, np.nan)
                    ).astype(np.float32)

        for col, spec in self.fitted.get("alternate_ids", {}).items():
            key = "|".join(self._text(row.values.get(c), c) for c in spec["components"])
            out[col] = np.array([spec["counts"].get(key, 0.0)], np.float32)

    def _text(self, value, col: str) -> str:
        """``build._as_text`` for one value, without building a frame."""
        if value is None or (not isinstance(value, str) and np.isnan(value)):
            return "__na__"
        return str(pd.Series([value], dtype=self.raw_dtypes[col]).astype("string").iloc[0])

    def _encoded(self, row: _Row, out) -> None:
        for col, spec in self.fitted.get("categoricals", {}).items():
            raw = row.values.get(col)
            value = "__missing__" if raw is None or (
                not isinstance(raw, str) and np.isnan(raw)) else self._text(raw, col)
            counts = spec["counts"]
            collapsed = value if value in counts else "__rare__"
            out[f"freq_{col}"] = np.array([counts.get(collapsed, 0.0)], np.float32)
            out[f"cat_{col}"] = np.array([self._categories[col].get(collapsed, -1)], np.int32)

    def _raw_numeric(self, row: _Row, t: int, out) -> None:
        day = np.array([t], np.int64) // build.SECONDS_PER_DAY
        for col in self.state["raw_numeric"]:
            values = np.array([row.values[col]]).astype(np.float32)
            if self.stationarise and col.startswith("D") and col[1:].isdigit():
                values = (day - values).astype(np.float32)
            out[col] = values

    # -- the stream ---------------------------------------------------------
    def update(self, raw: Mapping[str, Any] | pd.Series, label: int | None = None) -> int:
        """Append a transaction to the history. Returns its transaction id."""
        row = self.frame(raw)
        return self._append(row, self._codes(row), label)

    def score_and_update(
        self, raw: Mapping[str, Any] | pd.Series, label: int | None = None
    ) -> tuple[pd.DataFrame, int]:
        """Features as of arrival (training dtypes), then remember the transaction."""
        row = self.frame(raw)
        codes = self._codes(row)
        features = self._features(row, codes).astype(self.dtypes)
        return features, self._append(row, codes, label)

    def _append(self, row: _Row, codes: dict[str, int], label: int | None) -> int:
        arrival = _Arrival(txn_id=row.txn_id, t=row.t, codes=codes,
                           label=None if label is None else int(label))
        self.arrival_position[row.txn_id] = len(self.arrivals)
        self.arrivals.append(arrival)

        for entity, code in codes.items():
            live = self.live[entity].get(code)
            if live is None:
                live = self.live[entity][code] = _Live()
            live.times.append(row.t)
            live.cum_amt.append(live.cum_amt[-1] + row.amount)
            live.s2 += row.amount * row.amount
            if self.agg_on and entity in self.agg_entities:
                for col in self.value_cols:
                    v = float(np.float64(row.values[col]))
                    if not np.isnan(v):
                        acc = live.values.setdefault(col, [0.0, 0.0, 0.0])
                        acc[0] += 1
                        acc[1] += v
                        acc[2] += v * v
        for left, right in self.edges:
            code, partner = codes[left], codes[right]
            if not self._in_history(left, right, code, partner):
                self.live[left][code].partners.setdefault(right, set()).add(partner)
        if arrival.label is not None:
            self._add_label(arrival, arrival.label)
        self.latest_time = max(self.latest_time, row.t)
        self.max_id = max(self.max_id, row.txn_id)
        return row.txn_id

    def _add_label(self, arrival: _Arrival, label: int, sign: int = 1) -> None:
        for entity, code in arrival.codes.items():
            live = self.live[entity][code]
            live.label_n += sign
            live.label_sum += sign * label
            if self.label_delay:
                labels = np.diff(live.labelled_cum).tolist()
                if sign > 0:
                    k = bisect.bisect_right(live.labelled_t, arrival.t)
                    live.labelled_t.insert(k, arrival.t)
                    labels.insert(k, float(label))
                else:
                    k = live.labelled_t.index(arrival.t)
                    live.labelled_t.pop(k)
                    labels.pop(k)
                live.labelled_cum = [0.0] + np.cumsum(labels).tolist()

    def record_outcome(self, txn_id: int, label: int) -> bool:
        """Confirm the outcome of an arrival; later transactions will see it."""
        position = self.arrival_position.get(int(txn_id))
        if position is None:
            return False
        arrival = self.arrivals[position]
        if arrival.label is not None:
            self._add_label(arrival, arrival.label, sign=-1)
        arrival.label = int(label)
        self._add_label(arrival, arrival.label)
        return True

    def history_summary(self, raw: Mapping[str, Any] | pd.Series) -> dict:
        """What the store knows about the entities a transaction touches."""
        row = self.frame(raw)
        codes = self._codes(row)
        summary = {}
        for entity in ("card", "uid", "device", "email"):
            prior = self._degree(entity, codes[entity])
            summary[entity] = {"known": prior > 0, "prior_transactions": prior}
        return summary

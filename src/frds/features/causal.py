"""Causal aggregation primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_groups(values: pd.Series) -> np.ndarray:
    """Map arbitrary labels (including NaN) to dense non-negative int codes."""
    codes, _ = pd.factorize(values, use_na_sentinel=False)
    return codes.astype(np.int64)


def _packed_order(group_codes: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (order, packed_key_sorted, offset)."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    t = times[order]
    offset = int(times.max()) + 1
    packed = g * offset + t
    return order, packed, offset


def count_in_window(
    group_codes: np.ndarray, times: np.ndarray, window_seconds: int
) -> np.ndarray:
    """Prior transactions by the same entity within the trailing window."""
    order, packed, offset = _packed_order(group_codes, times)
    g = group_codes[order]
    t = times[order]
    lower = g * offset + np.maximum(t - window_seconds, 0)
    lo = np.searchsorted(packed, lower, side="left")
    counts = (np.arange(len(t), dtype=np.int64) - lo).astype(np.int32)

    out = np.empty_like(counts)
    out[order] = counts
    return out


def sum_in_window(
    group_codes: np.ndarray, times: np.ndarray, values: np.ndarray, window_seconds: int
) -> np.ndarray:
    """Sum of ``values`` over the same entity's prior rows in the window."""
    order, packed, offset = _packed_order(group_codes, times)
    g = group_codes[order]
    t = times[order]
    v = values[order].astype(np.float64)

    cumulative = np.concatenate([[0.0], np.cumsum(v)])
    lower = g * offset + np.maximum(t - window_seconds, 0)
    lo = np.searchsorted(packed, lower, side="left")
    hi = np.arange(len(t), dtype=np.int64)  # exclusive of the current row
    sums = (cumulative[hi] - cumulative[lo]).astype(np.float32)

    out = np.empty_like(sums)
    out[order] = sums
    return out


def seconds_since_previous(group_codes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Recency: gap to this entity's previous transaction, NaN on first sight."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    t = times[order].astype(np.float64)

    gap = np.full(len(t), np.nan, dtype=np.float32)
    same = np.empty(len(t), dtype=bool)
    same[0] = False
    same[1:] = g[1:] == g[:-1]
    gap[1:] = np.where(same[1:], t[1:] - t[:-1], np.nan)

    out = np.empty_like(gap)
    out[order] = gap
    return out


def _prior_moments(
    group_codes: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(order, prior count, prior sum, prior sum of squares) over observed values."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    v = values[order].astype(np.float64)
    observed = ~np.isnan(v)
    v0 = np.where(observed, v, 0.0)

    n = len(v)
    ccount = np.concatenate([[0], np.cumsum(observed)])
    csum = np.concatenate([[0.0], np.cumsum(v0)])
    csq = np.concatenate([[0.0], np.cumsum(v0 * v0)])
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    idx = np.arange(n)
    return (
        order,
        ccount[idx] - ccount[group_start],
        csum[idx] - csum[group_start],
        csq[idx] - csq[group_start],
    )


def moments_to_mean(count: np.ndarray, s1: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, s1 / np.maximum(count, 1), np.nan)


def moments_to_std(count: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / np.maximum(count, 1)
        var = s2 / np.maximum(count, 1) - mean * mean
        return np.where(count > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)


def expanding_mean(group_codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of the entity's prior observed values. NaN until there is one."""
    order, count, s1, _ = _prior_moments(group_codes, values)
    out = np.empty(len(order), dtype=np.float32)
    out[order] = moments_to_mean(count, s1).astype(np.float32)
    return out


def expanding_std(group_codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Population std of the entity's prior observed values. NaN with fewer than 2."""
    order, count, s1, s2 = _prior_moments(group_codes, values)
    out = np.empty(len(order), dtype=np.float32)
    out[order] = moments_to_std(count, s1, s2).astype(np.float32)
    return out


def seconds_since_first_seen(group_codes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Age of the entity at this transaction: now minus its first sighting."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    t = times[order].astype(np.float64)

    n = len(t)
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    ages = (t - t[group_start]).astype(np.float32)

    out = np.empty(n, dtype=np.float32)
    out[order] = ages
    return out


def cumulative_count(group_codes: np.ndarray) -> np.ndarray:
    """Number of prior transactions by the entity, over all history."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    n = len(g)
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    counts = (np.arange(n) - group_start).astype(np.int32)

    out = np.empty(n, dtype=np.int32)
    out[order] = counts
    return out


def distinct_partners_so_far(
    left_codes: np.ndarray, right_codes: np.ndarray
) -> np.ndarray:
    """How many *distinct* right-entities this left-entity has touched before."""
    n = len(left_codes)
    pairs = pd.MultiIndex.from_arrays([left_codes, right_codes])
    # A pair contributes on its first occurrence in time order only.
    pair_first = (~pairs.duplicated(keep="first")).astype(np.int8)

    order = np.argsort(left_codes, kind="stable")
    g = left_codes[order]
    f = pair_first[order].astype(np.int64)
    csum = np.concatenate([[0], np.cumsum(f)])
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    totals = (csum[np.arange(n)] - csum[group_start]).astype(np.int32)

    out = np.empty(n, dtype=np.int32)
    out[order] = totals
    return out


def expanding_target_rate(
    group_codes: np.ndarray,
    target: np.ndarray,
    prior_weight: float = 20.0,
    times: np.ndarray | None = None,
    label_delay_seconds: int = 0,
    global_rate: float | None = None,
) -> np.ndarray:
    """Smoothed fraud rate among the entity's *known-outcome* prior transactions."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    y = target[order].astype(np.float64)

    n = len(y)
    csum = np.concatenate([[0.0], np.cumsum(y)])
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    idx = np.arange(n)

    if label_delay_seconds > 0:
        if times is None:
            raise ValueError("label_delay_seconds requires the time column")
        t = times[order]
        offset = int(times.max()) + 1
        packed = g * offset + t
        cutoff = g * offset + np.maximum(t - label_delay_seconds, 0)
        # side="right": a label is known once its transaction is at or before
        # the cutoff. Capped at the current row so it can never see itself.
        known = np.minimum(np.searchsorted(packed, cutoff, side="right"), idx)
        known = np.maximum(known, group_start)
    else:
        known = idx

    count = known - group_start
    prior_sum = csum[known] - csum[group_start]

    if global_rate is None:
        global_rate = float(target.mean())
    rates = (prior_sum + prior_weight * global_rate) / (count + prior_weight)

    out = np.empty(n, dtype=np.float32)
    out[order] = rates.astype(np.float32)
    return out

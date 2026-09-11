"""Causal aggregation primitives.

Every aggregate in this project is computed from rows that are strictly
*earlier* than the row being scored. That is the difference between a feature
a production scorer could actually have and a feature that quietly encodes the
answer. All four helpers here take a time-sorted frame and return arrays
aligned to it.

The window helpers avoid a per-group Python loop by packing (group, time) into
a single monotonically increasing key and doing one global ``searchsorted``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_groups(values: pd.Series) -> np.ndarray:
    """Map arbitrary labels (including NaN) to dense non-negative int codes."""
    codes, _ = pd.factorize(values, use_na_sentinel=False)
    return codes.astype(np.int64)


def _packed_order(group_codes: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (order, packed_key_sorted, offset).

    ``order`` sorts rows by (group, time); because the caller guarantees the
    frame is already time-sorted, a stable sort on the group code alone
    preserves time order inside each group.
    """
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    t = times[order]
    offset = int(times.max()) + 1
    packed = g * offset + t
    return order, packed, offset


def count_in_window(
    group_codes: np.ndarray, times: np.ndarray, window_seconds: int
) -> np.ndarray:
    """Prior transactions by the same entity within the trailing window.

    The current row is excluded, so the value is what a scorer would know at
    the instant the transaction arrives.
    """
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


def expanding_mean(group_codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of the entity's prior values. NaN until the entity has history."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    v = values[order].astype(np.float64)

    n = len(v)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    # Position within group.
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    pos = np.arange(n) - group_start

    prior_sum = csum[np.arange(n)] - csum[group_start]
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(pos > 0, prior_sum / np.maximum(pos, 1), np.nan)

    out = np.empty(n, dtype=np.float32)
    out[order] = means.astype(np.float32)
    return out


def expanding_std(group_codes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Population std of the entity's prior values. NaN with fewer than 2."""
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]
    v = values[order].astype(np.float64)

    n = len(v)
    csum = np.concatenate([[0.0], np.cumsum(v)])
    csq = np.concatenate([[0.0], np.cumsum(v * v)])
    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    pos = np.arange(n) - group_start

    idx = np.arange(n)
    s1 = csum[idx] - csum[group_start]
    s2 = csq[idx] - csq[group_start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / np.maximum(pos, 1)
        var = s2 / np.maximum(pos, 1) - mean * mean
        stds = np.where(pos > 1, np.sqrt(np.maximum(var, 0.0)), np.nan)

    out = np.empty(n, dtype=np.float32)
    out[order] = stds.astype(np.float32)
    return out


def seconds_since_first_seen(group_codes: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Age of the entity at this transaction: now minus its first sighting.

    The denominator for turning a cumulative counter into a rate. A device with
    500 prior transactions means something entirely different in its first week
    than in its sixth month, and a raw count cannot express that difference.
    """
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
    """How many *distinct* right-entities this left-entity has touched before.

    The relational feature that matters for ring detection: a device that has
    already been seen with eleven different cards is not a normal device. It is
    computed exactly, not approximately - a pair contributes on its first
    occurrence only, and the running total is shifted so the current row is
    excluded.
    """
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
) -> np.ndarray:
    """Smoothed fraud rate among the entity's *known-outcome* prior transactions.

    Target encoding done causally: row i sees only labels from rows that
    precede it for the same entity, shrunk towards the global prior so a single
    early fraud on a new card does not produce a rate of 1.0.

    ``label_delay_seconds`` models the fact that fraud labels are not available
    the moment the fraud happens. A chargeback takes weeks to land, so a scorer
    running at time t knows outcomes only for transactions at or before
    ``t - delay``. With the default of zero this reduces to "every earlier
    transaction's label is already known", which is the optimistic assumption
    stated in section 4.3 of the report; setting a realistic delay is how that
    assumption gets tested rather than asserted.
    """
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

    global_rate = float(target.mean())
    rates = (prior_sum + prior_weight * global_rate) / (count + prior_weight)

    out = np.empty(n, dtype=np.float32)
    out[order] = rates.astype(np.float32)
    return out

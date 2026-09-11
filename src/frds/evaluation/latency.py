"""Governance layer (L5): serving cost.

Report objective 6 and gap G6 - inference latency is measured, not assumed.
What is timed is the per-transaction path a live scorer would run: base model
inference, meta-learner fusion, calibration and tier assignment. Feature
computation is timed separately, because in production the velocity counters
come from an online cache rather than being recomputed per request.

Batch-of-one timing is deliberately included. A model that scores 100k rows in
two seconds can still take 40 ms per single-row call, and the single-row number
is the one that decides whether real-time interdiction is feasible.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("frds.evaluation.latency")


def _percentiles(samples_ms: np.ndarray) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(samples_ms, 50)), 3),
        "p90_ms": round(float(np.percentile(samples_ms, 90)), 3),
        "p95_ms": round(float(np.percentile(samples_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(samples_ms, 99)), 3),
        "mean_ms": round(float(np.mean(samples_ms)), 3),
        "max_ms": round(float(np.max(samples_ms)), 3),
    }


def measure_single_row(
    score_fn, X: pd.DataFrame, n_samples: int = 500, warmup: int = 20
) -> dict[str, float]:
    """Time ``score_fn`` on one row at a time."""
    n = min(n_samples, len(X))
    idx = np.random.default_rng(0).choice(len(X), size=n, replace=False)

    for i in idx[:warmup]:
        score_fn(X.iloc[[i]])

    samples = np.empty(n, dtype=np.float64)
    for k, i in enumerate(idx):
        row = X.iloc[[i]]
        start = time.perf_counter()
        score_fn(row)
        samples[k] = (time.perf_counter() - start) * 1000.0

    stats = _percentiles(samples)
    stats["mode"] = "single_row"
    stats["samples"] = int(n)
    log.info("single-row latency: p50 %.2f ms, p95 %.2f ms", stats["p50_ms"], stats["p95_ms"])
    return stats


def measure_batch(
    score_fn, X: pd.DataFrame, batch_size: int = 1024, n_batches: int = 20
) -> dict[str, float]:
    """Amortised per-transaction cost when scoring in batches."""
    n_batches = min(n_batches, max(1, len(X) // batch_size))
    samples = np.empty(n_batches, dtype=np.float64)

    score_fn(X.iloc[:batch_size])  # warm up
    for k in range(n_batches):
        chunk = X.iloc[k * batch_size: (k + 1) * batch_size]
        if chunk.empty:
            samples = samples[:k]
            break
        start = time.perf_counter()
        score_fn(chunk)
        samples[k] = (time.perf_counter() - start) * 1000.0 / len(chunk)

    stats = _percentiles(samples)
    stats["mode"] = f"batch_{batch_size}"
    stats["batches"] = int(len(samples))
    stats["throughput_tps"] = round(1000.0 / max(stats["mean_ms"], 1e-9), 1)
    log.info(
        "batch-%d latency: %.4f ms/txn amortised (%.0f txn/s)",
        batch_size, stats["mean_ms"], stats["throughput_tps"],
    )
    return stats


def latency_report(score_fn, X: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    """Single-row and batched timings side by side."""
    subset = X.iloc[: min(sample_size, len(X))]
    rows = [
        measure_single_row(score_fn, subset, n_samples=min(500, len(subset))),
        measure_batch(score_fn, subset, batch_size=256),
        measure_batch(score_fn, subset, batch_size=1024),
    ]
    frame = pd.DataFrame(rows)
    cols = ["mode", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"]
    extra = [c for c in frame.columns if c not in cols]
    return frame[cols + extra]

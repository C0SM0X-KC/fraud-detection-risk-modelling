"""Governance layer (L5): serving cost."""

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


# ------------------------------------------------------------ product path --
def product_latency(
    service,
    raws: list[dict],
    target_ms: float,
    n_single: int = 500,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Time the product path on raw transactions, in arrival order."""
    rows = []
    stage_keys = ("features_ms", "model_ms", "decision_ms", "explanation_ms")

    def run(mode: str, explain: bool, chunk: list[dict]) -> None:
        for raw in chunk[:20]:                       # warm caches, not timed
            service.score(raw, explain=explain, remember=False)
        totals, stages = [], {k: [] for k in stage_keys}
        for raw in chunk:
            start = time.perf_counter()
            decision = service.score(raw, explain=explain)
            totals.append((time.perf_counter() - start) * 1000.0)
            for k in stage_keys:
                if k in decision["timings_ms"]:
                    stages[k].append(decision["timings_ms"][k])
        stats = _percentiles(np.asarray(totals))
        stats["mode"] = mode
        stats["samples"] = len(totals)
        for k, values in stages.items():
            if values:
                stats[f"{k.removesuffix('_ms')}_p50_ms"] = round(float(np.median(values)), 3)
        stats["target_ms"] = target_ms
        stats["meets_target_p95"] = bool(stats["p95_ms"] <= target_ms)
        rows.append(stats)
        log.info("%s: p50 %.2f ms, p95 %.2f ms (target %.0f ms)",
                 mode, stats["p50_ms"], stats["p95_ms"], target_ms)

    half = max(1, min(n_single, len(raws) // 3))
    run("single_transaction", False, raws[:half])
    run("single_transaction_with_reasons", True, raws[half: 2 * half])

    rest = raws[2 * half:]
    samples = []
    for k in range(0, len(rest) - batch_size + 1, batch_size):
        chunk = rest[k: k + batch_size]
        start = time.perf_counter()
        service.score_many(chunk)
        samples.append((time.perf_counter() - start) * 1000.0 / len(chunk))
        if len(samples) >= 10:
            break
    if samples:
        stats = _percentiles(np.asarray(samples))
        stats["mode"] = f"micro_batch_{batch_size}_per_txn"
        stats["samples"] = len(samples)
        stats["throughput_tps"] = round(1000.0 / max(stats["mean_ms"], 1e-9), 1)
        stats["target_ms"] = target_ms
        stats["meets_target_p95"] = bool(stats["p95_ms"] <= target_ms)
        rows.append(stats)

    frame = pd.DataFrame(rows)
    cols = ["mode", "p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms", "target_ms",
            "meets_target_p95"]
    return frame[cols + [c for c in frame.columns if c not in cols]]

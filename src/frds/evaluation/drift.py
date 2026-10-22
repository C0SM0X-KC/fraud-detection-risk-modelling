"""Governance layer (L5): concept-drift monitoring by Population Stability Index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..data.split import temporal_windows
from ..utils import get_logger

log = get_logger("frds.evaluation.drift")

SECONDS_PER_DAY = 86_400


def psi(
    reference: np.ndarray, actual: np.ndarray, n_bins: int = 10, epsilon: float = 1e-6
) -> float:
    """PSI of ``actual`` against ``reference`` using reference quantile bins."""
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    reference = reference[np.isfinite(reference)]
    actual = actual[np.isfinite(actual)]
    if reference.size == 0 or actual.size == 0:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 3:
        return 0.0
    inner = edges[1:-1]

    ref_counts = np.bincount(np.digitize(reference, inner), minlength=edges.size - 1)
    act_counts = np.bincount(np.digitize(actual, inner), minlength=edges.size - 1)

    ref_pct = np.maximum(ref_counts / ref_counts.sum(), epsilon)
    act_pct = np.maximum(act_counts / act_counts.sum(), epsilon)

    return float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))


def interpret(value: float, alert_threshold: float = 0.25) -> str:
    if not np.isfinite(value):
        return "undefined"
    if value < 0.10:
        return "stable"
    if value < alert_threshold:
        return "moderate"
    return "significant"


def score_drift(
    scores: np.ndarray, times: np.ndarray, cfg: Config
) -> pd.DataFrame:
    """PSI of the model's own score distribution across temporal windows."""
    n_windows = int(cfg.require("evaluation.psi_windows"))
    n_bins = int(cfg.require("evaluation.psi_bins"))
    alert = float(cfg.require("evaluation.psi_alert_threshold"))

    windows = temporal_windows(np.asarray(times), n_windows)
    reference = scores[windows[0]]

    rows = []
    for k, idx in enumerate(windows):
        if idx.size == 0:
            continue
        value = 0.0 if k == 0 else psi(reference, scores[idx], n_bins)
        rows.append(
            {
                "window": k,
                "rows": int(idx.size),
                "day_start": round(float(times[idx].min()) / SECONDS_PER_DAY, 1),
                "day_end": round(float(times[idx].max()) / SECONDS_PER_DAY, 1),
                "mean_score": round(float(scores[idx].mean()), 6),
                "psi_vs_window0": round(value, 5),
                "status": "reference" if k == 0 else interpret(value, alert),
            }
        )
    frame = pd.DataFrame(rows)
    worst = frame["psi_vs_window0"].max()
    log.info("score PSI across %d windows, worst = %.4f (%s)",
             len(frame), worst, interpret(worst, alert))
    return frame


def feature_drift(
    X: pd.DataFrame, times: np.ndarray, cfg: Config, top_n: int = 25
) -> pd.DataFrame:
    """PSI per feature between the first and last temporal window."""
    n_windows = int(cfg.require("evaluation.psi_windows"))
    n_bins = int(cfg.require("evaluation.psi_bins"))
    alert = float(cfg.require("evaluation.psi_alert_threshold"))

    windows = temporal_windows(np.asarray(times), n_windows)
    ref_idx, last_idx = windows[0], windows[-1]

    rows = []
    for col in X.columns:
        ref = X[col].to_numpy()[ref_idx]
        act = X[col].to_numpy()[last_idx]
        value = psi(ref, act, n_bins)
        rows.append(
            {
                "feature": col,
                "psi_first_vs_last": round(value, 5) if np.isfinite(value) else np.nan,
                "status": interpret(value, alert),
            }
        )

    frame = (
        pd.DataFrame(rows)
        .sort_values("psi_first_vs_last", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    n_alert = int((frame["status"] == "significant").sum())
    log.info("feature drift: %d of %d features above the alert threshold",
             n_alert, len(frame))
    return frame.head(top_n)


def prevalence_drift(
    y_true: np.ndarray, times: np.ndarray, cfg: Config
) -> pd.DataFrame:
    """Observed fraud rate per window - the label-side companion to score PSI."""
    n_windows = int(cfg.require("evaluation.psi_windows"))
    windows = temporal_windows(np.asarray(times), n_windows)
    base = float(y_true[windows[0]].mean()) if windows[0].size else float("nan")

    rows = []
    for k, idx in enumerate(windows):
        if idx.size == 0:
            continue
        rate = float(y_true[idx].mean())
        rows.append(
            {
                "window": k,
                "rows": int(idx.size),
                "fraud_rate": round(rate, 5),
                "relative_change_vs_window0": round(rate / base - 1, 4)
                if base > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)

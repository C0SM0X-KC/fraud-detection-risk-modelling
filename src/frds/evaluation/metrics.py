"""Evaluation metrics (report Table 4.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..utils import get_logger

log = get_logger("frds.evaluation.metrics")


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> dict[str, float]:
    """Recall achievable while holding the false-positive rate at or below the
    target, plus the score threshold that achieves it."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    feasible = np.flatnonzero(fpr <= target_fpr)
    if feasible.size == 0:
        return {"recall": 0.0, "threshold": float("nan"), "achieved_fpr": 0.0}
    best = feasible[-1]  # the ROC curve is monotone, so the last one is the best
    return {
        "recall": float(tpr[best]),
        "threshold": float(thresholds[best]),
        "achieved_fpr": float(fpr[best]),
    }


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_fraction: float) -> dict[str, float]:
    """Precision and recall when reviewing the top ``k_fraction`` of traffic."""
    n = len(scores)
    k = max(1, int(round(k_fraction * n)))
    top = np.argsort(scores)[::-1][:k]
    caught = int(np.sum(y_true[top]))
    total = int(np.sum(y_true))
    return {
        "k_fraction": k_fraction,
        "alerts": k,
        "precision": caught / k,
        "recall": caught / max(total, 1),
    }


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(
            (precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0
        )
    best = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    return {
        "f1": float(f1[best]),
        "threshold": float(thresholds[best]) if len(thresholds) else 0.5,
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    fpr_targets: list[float] | None = None,
    label: str = "model",
) -> dict:
    """The full metric block for one score vector."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    fpr_targets = fpr_targets or [0.001, 0.005, 0.01, 0.05]

    prevalence = float(y_true.mean())
    best_f1 = best_f1_threshold(y_true, scores)
    hard = (scores >= best_f1["threshold"]).astype(int)

    out: dict = {
        "label": label,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "prevalence": prevalence,
        "auc_pr": float(average_precision_score(y_true, scores)),
        # The no-skill AUC-PR equals the prevalence; a lift below 1.0 means the
        # model is worse than guessing at the base rate.
        "auc_pr_lift_over_prevalence": float(
            average_precision_score(y_true, scores) / max(prevalence, 1e-12)
        ),
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "f1_best": best_f1["f1"],
        "f1_threshold": best_f1["threshold"],
        "precision_at_best_f1": best_f1["precision"],
        "recall_at_best_f1": best_f1["recall"],
        "mcc": float(matthews_corrcoef(y_true, hard)),
        "brier": float(brier_score_loss(y_true, np.clip(scores, 0, 1))),
        "accuracy_not_a_headline_metric": float((hard == y_true).mean()),
        "accuracy_of_always_legitimate": float(1 - prevalence),
    }

    for target in fpr_targets:
        stats = recall_at_fpr(y_true, scores, target)
        out[f"recall_at_fpr_{target}"] = stats["recall"]
        out[f"threshold_at_fpr_{target}"] = stats["threshold"]

    for k in (0.001, 0.005, 0.01, 0.05):
        stats = precision_at_k(y_true, scores, k)
        out[f"precision_at_top_{k}"] = stats["precision"]
        out[f"recall_at_top_{k}"] = stats["recall"]

    return out


def comparison_table(results: list[dict]) -> pd.DataFrame:
    """One row per model, ordered by the primary metric."""
    keep = [
        "label", "auc_pr", "auc_pr_lift_over_prevalence", "auc_roc", "f1_best", "mcc",
        "brier", "recall_at_fpr_0.001", "recall_at_fpr_0.01", "recall_at_top_0.005",
    ]
    frame = pd.DataFrame(results)
    cols = [c for c in keep if c in frame.columns]
    return frame[cols].sort_values("auc_pr", ascending=False).reset_index(drop=True)


def confusion_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, int]:
    pred = scores >= threshold
    y = np.asarray(y_true).astype(bool)
    return {
        "tp": int(np.sum(pred & y)),
        "fp": int(np.sum(pred & ~y)),
        "fn": int(np.sum(~pred & y)),
        "tn": int(np.sum(~pred & ~y)),
    }

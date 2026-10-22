"""Figures for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .utils import get_logger  # noqa: E402

log = get_logger("frds.plots")

DPI = 150
FIGSIZE = (7.0, 4.4)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)
    return path


def precision_recall_curves(
    y_true: np.ndarray, score_map: dict[str, np.ndarray], path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, scores in score_map.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y_true, scores)
        ax.plot(recall, precision, linewidth=1.6, label=f"{name} (AUC-PR {ap:.4f})")

    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, linestyle="--", linewidth=1.0, color="grey",
               label=f"no-skill = prevalence ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the held-out test span")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def roc_curves(y_true: np.ndarray, score_map: dict[str, np.ndarray], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for name, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, linewidth=1.6, label=name)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("False-positive rate (log scale)")
    ax.set_ylabel("Recall")
    ax.set_title("ROC, log-scaled FPR - the operating region a fraud team lives in")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def reliability_diagram(curve: dict[str, list[float]], ece: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1.0, label="perfect calibration")
    ax.plot(curve["predicted"], curve["observed"], "o-", linewidth=1.6, markersize=4,
            label=f"calibrated score (ECE {ece:.5f})")
    limit = max(max(curve["predicted"], default=0.1), max(curve["observed"], default=0.1)) * 1.1
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Reliability of the calibrated risk score")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def score_distribution(
    scores: np.ndarray, y_true: np.ndarray, medium_lo: int, high_lo: int, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.arange(0, 102, 2)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.65, label="legitimate", log=True)
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.75, label="fraud", log=True)
    ax.axvline(medium_lo, linestyle="--", color="darkorange", linewidth=1.2,
               label=f"medium >= {medium_lo}")
    ax.axvline(high_lo, linestyle="--", color="firebrick", linewidth=1.2,
               label=f"high >= {high_lo}")
    ax.set_xlabel("Risk score (0-100)")
    ax.set_ylabel("Transactions (log scale)")
    ax.set_title("Risk-score distribution and operating tier boundaries")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def psi_over_windows(frame: pd.DataFrame, alert_threshold: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(frame["window"], frame["psi_vs_window0"], width=0.6)
    ax.axhline(0.10, linestyle="--", color="goldenrod", linewidth=1.1, label="0.10 watch")
    ax.axhline(alert_threshold, linestyle="--", color="firebrick", linewidth=1.1,
               label=f"{alert_threshold:.2f} retrain")
    ax.set_xlabel("Temporal window (window 0 = reference)")
    ax.set_ylabel("PSI vs window 0")
    ax.set_title("Score drift across the test timeline")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)


def model_comparison(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ordered = frame.sort_values("auc_pr")
    ax.barh(ordered["label"], ordered["auc_pr"])
    for y, (value, label) in enumerate(zip(ordered["auc_pr"], ordered["label"])):
        ax.text(value, y, f" {value:.4f}", va="center", fontsize=8)
    ax.set_xlabel("AUC-PR on the held-out test span")
    ax.set_title("Model comparison under one identical protocol")
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)


def shap_importance(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ordered = frame.sort_values("mean_abs_shap")
    ax.barh(ordered["feature"], ordered["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature attribution over high-risk alerts")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)


def shap_groups(frame: pd.DataFrame, path: Path) -> Path:
    """Attribution by evidence family: which kinds of signal decide alerts."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ordered = frame.sort_values("mean_abs_shap")
    ax.barh(ordered["group"], ordered["mean_abs_shap"])
    for y, (value, share) in enumerate(zip(ordered["mean_abs_shap"], ordered["share"])):
        ax.text(value, y, f"  {share:.0%}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value| summed within the family")
    ax.set_title("Attribution by evidence family over high-risk alerts")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)


def shap_beeswarm(values: np.ndarray, X: pd.DataFrame, path: Path, max_display: int = 15) -> Path:
    """Per-alert attributions coloured by feature value (direction, not just size)."""
    import shap

    shap.summary_plot(values, X, max_display=max_display, show=False, plot_size=(7.0, 5.6))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP values over high-risk alerts")
    return _save(fig, path)


def faithfulness(frame: pd.DataFrame, path: Path) -> Path:
    """Score drop when an alert's top reasons are removed vs random features."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    k = frame["k"].astype(str)
    width = 0.38
    x = np.arange(len(frame))
    ax.bar(x - width / 2, frame["mean_score_drop_top"], width, label="top-k SHAP features removed")
    ax.bar(x + width / 2, frame["mean_score_drop_random"], width, label="k random features removed")
    ax.set_xticks(x, [f"k = {v}" for v in k])
    ax.set_ylabel("Mean fall in risk score (0-100)")
    ax.set_title("Explanation faithfulness: deletion test on high-risk alerts")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)


def robustness(frame: pd.DataFrame, path: Path) -> Path:
    """Evasion rate against attack budget: targeted PGD vs random noise."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for attack, style in (("pgd", "-o"), ("random", "--s")):
        part = frame[frame["attack"] == attack]
        label = "PGD (transfer from surrogate)" if attack == "pgd" else "random noise, same budget"
        ax.plot(part["epsilon_std"], 100 * part["evasion_rate"], style, label=label)
    ax.set_xlabel("Perturbation budget (standard deviations per mutable feature)")
    ax.set_ylabel("Caught frauds that now clear (%)")
    ax.set_title("Evasion of the deployed model under attacker-controlled features")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return _save(fig, path)

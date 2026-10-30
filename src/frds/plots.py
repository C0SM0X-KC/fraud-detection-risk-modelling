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


# ---------------------------------------------------- data analytics ----
SPLIT_COLOURS = {"train": "#4C72B0", "validation": "#DD8452", "test": "#55A868"}
ORDERS = {
    "Account history": ["first seen", "1-4 earlier", "5-19 earlier", "20+ earlier"],
    "Amount band": ["under 25", "25-50", "50-100", "100-250", "250-500", "500+"],
}


def eda_timeline(daily: pd.DataFrame, path: Path) -> Path:
    """Daily volume by span, with the daily fraud rate on a second axis."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for span, colour in SPLIT_COLOURS.items():
        part = daily[daily["split"] == span]
        ax.bar(part["day"], part["transactions"], width=1.0, color=colour, alpha=0.55,
               label=f"{span} transactions")
    ax.set_xlabel("Day of the dataset")
    ax.set_ylabel("Transactions per day")
    rate = ax.twinx()
    smooth = daily["fraud_rate"].rolling(7, center=True, min_periods=1).mean()
    rate.plot(daily["day"], 100 * smooth, color="firebrick", linewidth=1.5,
              label="fraud rate, 7-day average")
    rate.set_ylabel("Fraud rate (%)")
    rate.set_ylim(0, max(100 * smooth.max() * 1.3, 1))
    handles = ax.get_legend_handles_labels()[0] + rate.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    ax.set_title("Daily transaction volume and fraud rate across the timeline")
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)


def eda_missingness(quality: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ordered = quality.sort_values("mean_missing_pct")
    ax.barh(ordered["family"], ordered["mean_missing_pct"], label="mean over the family")
    ax.scatter(ordered["max_missing_pct"], ordered["family"], color="firebrick", zorder=3,
               s=18, label="worst column")
    for y, (value, worst, n) in enumerate(zip(ordered["mean_missing_pct"],
                                              ordered["max_missing_pct"], ordered["columns"])):
        ax.text(max(value, worst), y, f"   {value:.0f}%  ({n} cols)", va="center", fontsize=8)
    ax.set_xlim(0, 125)
    ax.set_xlabel("Missing values (% of training transactions)")
    ax.set_title("Missing values by column family")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)


def eda_amount(train: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.logspace(np.log10(0.25), np.log10(train["TransactionAmt"].max() + 1), 60)
    for label, value, colour in (("legitimate", 0, "#4C72B0"), ("fraud", 1, "firebrick")):
        amounts = train.loc[train["isFraud"] == value, "TransactionAmt"]
        ax.hist(amounts, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                color=colour, label=f"{label} (median {amounts.median():.0f})")
        ax.axvline(amounts.median(), color=colour, linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("Transaction amount (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("Transaction amount: fraud against legitimate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def _rate_bars(ax, table: pd.DataFrame, overall: float, title: str, horizontal: bool = False):
    rate = 100 * table["fraud_rate"]
    err = np.clip([rate - 100 * table["ci_low"], 100 * table["ci_high"] - rate], 0, None)
    if horizontal:
        ax.barh(table["level"], rate, xerr=err, capsize=2, color="#4C72B0", alpha=0.8)
        ax.axvline(100 * overall, linestyle="--", color="grey", linewidth=1.0)
        for y, share in enumerate(table["share_of_volume"]):
            ax.text(0, y, f" {share:.0%}", va="center", fontsize=7, color="white")
        ax.set_xlabel("Fraud rate (%)")
        ax.grid(alpha=0.25, axis="x")
    else:
        ax.bar(table["level"], rate, yerr=err, capsize=2, color="#4C72B0", alpha=0.8)
        ax.axhline(100 * overall, linestyle="--", color="grey", linewidth=1.0)
        ax.set_ylabel("Fraud rate (%)")
        ax.grid(alpha=0.25, axis="y")
    ax.set_title(title, fontsize=10)


def eda_time_cycles(hourly: pd.DataFrame, weekly: pd.DataFrame, overall: float,
                    path: Path) -> Path:
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.8),
                                      gridspec_kw={"width_ratios": [3, 1.3]})
    for ax, table, title in ((left, hourly, "By hour of day"),
                             (right, weekly, "By day of the weekly cycle")):
        _rate_bars(ax, table, overall, title)
        volume = ax.twinx()
        volume.plot(table["level"], table["transactions"] / 1000, color="black",
                    linewidth=1.2, marker=".")
        volume.set_ylabel("Transactions (thousands)", fontsize=8)
        volume.tick_params(labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
    left.set_xlabel("Hour (dataset clock)")
    right.set_xlabel("Day index")
    fig.suptitle("Fraud rate (bars, 95% interval) and volume (line) through the day and week",
                 fontsize=10)
    return _save(fig, path)


def eda_categories(rates: pd.DataFrame, overall: float, path: Path) -> Path:
    panels = ["Product code", "Card network", "Card type", "Device type",
              "Purchaser e-mail domain", "Amount band"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.0))
    for ax, dimension in zip(axes.ravel(), panels):
        table = rates[(rates["dimension"] == dimension) & (rates["transactions"] >= 1000)]
        if dimension in ORDERS:
            table = table.set_index("level").reindex(ORDERS[dimension]).dropna().reset_index()
        else:
            table = table.sort_values("fraud_rate")
        _rate_bars(ax, table, overall, dimension, horizontal=True)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"Fraud rate by segment, training span (dashed: overall {overall:.2%}; "
                 "white: share of volume; segments under 1,000 transactions omitted)", fontsize=10)
    return _save(fig, path)


def eda_correlation(corr: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    image = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Spearman correlation")
    ax.set_title("Correlation between amount, count (C) and time-delta (D) features")
    return _save(fig, path)


def eda_pareto(curve: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(100 * curve["share_of_fraud_accounts"], 100 * curve["share_of_fraud_value"],
            linewidth=1.8, label="accounts ranked by fraud value")
    ax.plot([0, 100], [0, 100], "--", color="grey", linewidth=1.0, label="equal shares")
    k = int(np.searchsorted(curve["share_of_fraud_value"].to_numpy(), 0.8))
    x80 = 100 * curve["share_of_fraud_accounts"].iloc[min(k, len(curve) - 1)]
    ax.axhline(80, color="firebrick", linewidth=0.9, linestyle=":")
    ax.axvline(x80, color="firebrick", linewidth=0.9, linestyle=":")
    ax.text(x80, 4, f"  {x80:.0f}% of accounts -> 80% of value", fontsize=8, color="firebrick")
    ax.set_xlabel("Accounts with fraud, largest first (%)")
    ax.set_ylabel("Cumulative share of fraud value (%)")
    ax.set_title("Concentration of fraud value across accounts (Pareto)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.25)
    return _save(fig, path)


def effect_sizes(categorical: pd.DataFrame, numeric: pd.DataFrame, path: Path) -> Path:
    """How strongly each variable separates fraud from legitimate, on one scale."""
    from matplotlib.patches import Patch

    kinds = {"category (Cramer's V)": "#4C72B0", "numeric (|rank-biserial r|)": "#DD8452"}
    rows = pd.concat([
        pd.DataFrame({"variable": categorical["variable"], "effect": categorical["cramers_v"],
                      "kind": "category (Cramer's V)"}),
        pd.DataFrame({"variable": numeric["variable"], "effect": numeric["rank_biserial"].abs(),
                      "kind": "numeric (|rank-biserial r|)"}),
    ]).sort_values("effect")
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.barh(rows["variable"], rows["effect"], color=rows["kind"].map(kinds))
    for x, name in ((0.1, "small"), (0.3, "medium"), (0.5, "large")):
        ax.axvline(x, linestyle="--", color="grey", linewidth=0.8)
        ax.text(x, len(rows) - 0.4, f" {name}", fontsize=7, color="grey")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in kinds.items()],
              fontsize=8, loc="lower right")
    ax.set_xlabel("Effect size")
    ax.set_title("Strength of association with fraud, training span")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.25, axis="x")
    return _save(fig, path)


def segment_detection(segments: pd.DataFrame, path: Path) -> Path:
    """Share of fraud caught and precision of the high tier, segment by segment."""
    panels = ["Account history", "Amount band", "Product code", "Device type"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2))
    for ax, dimension in zip(axes.ravel(), panels):
        part = segments[(segments["dimension"] == dimension) & (segments["fraud"] >= 20)]
        if dimension in ORDERS:
            part = part.set_index("segment").reindex(ORDERS[dimension]).dropna(
                subset=["fraud"]).reset_index()
        x = np.arange(len(part))
        ax.bar(x - 0.2, 100 * part["fraud_caught_flagged"], 0.4,
               label="fraud caught (medium + high)")
        ax.bar(x + 0.2, 100 * part["precision_high"], 0.4, label="precision of the high tier")
        ax.set_xticks(x, part["segment"], fontsize=7, rotation=20)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title(dimension, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Detection by segment on the held-out test span (segments with 20+ frauds)",
                 fontsize=10)
    return _save(fig, path)

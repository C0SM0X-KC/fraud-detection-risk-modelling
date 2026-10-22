"""Ablation: how much of the headline result depends on knowing outcomes early?

    python scripts/ablation_label_delay.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from frds import pipeline                                      # noqa: E402
from frds.config import load_config                            # noqa: E402
from frds.evaluation import metrics                            # noqa: E402
from frds.models.train import load_scores                      # noqa: E402
from frds.utils import get_logger, setup_logging, write_json   # noqa: E402

log = get_logger("frds.ablation")

VARIANTS: list[tuple[str, dict]] = [
    ("delay_0", {"features.label_delay_days": 0}),
    ("delay_30", {"features.label_delay_days": 30}),
    ("delay_60", {"features.label_delay_days": 60}),
    ("none", {"features.disable_target_encoding": True}),
]

MODELS = ["xgboost", "lightgbm"]
OUT_ROOT = REPO_ROOT / "reports" / "ablation_label_delay"


def run_variant(name: str, overrides: dict) -> list[dict]:
    root = OUT_ROOT / name
    cfg = load_config()
    for kind in ("processed", "artifacts"):
        cfg.override(f"paths.{kind}", str(root / kind))
    cfg.override("paths.reports", str(root / "reports"))
    cfg.ensure_dirs()
    for sub in ("figures", "tables", "metrics"):
        (root / "reports" / sub).mkdir(parents=True, exist_ok=True)

    cfg.override("models.enabled", MODELS)
    cfg.override("split.n_oof_folds", 2)
    for key, value in overrides.items():
        cfg.override(key, value)

    # Ingestion does not depend on the variant, so all variants share merged.parquet.
    log.info("=== variant %s: %s ===", name, overrides)
    started = time.perf_counter()
    cpu_started = time.process_time()
    pipeline.stage_features(cfg)
    pipeline.stage_train(cfg)

    scores = load_scores(cfg)
    y_test = scores["y_test"]
    fpr_targets = list(cfg.require("evaluation.fpr_targets"))

    rows = []
    for model in MODELS:
        key = f"test__{model}"
        if key not in scores:
            log.warning("%s produced no scores for %s", name, model)
            continue
        result = metrics.evaluate(y_test, scores[key], fpr_targets, label=model)
        rows.append(
            {
                "variant": name,
                "model": model,
                "auc_pr": round(result["auc_pr"], 4),
                "auc_roc": round(result["auc_roc"], 4),
                "f1_best": round(result["f1_best"], 4),
                "mcc": round(result["mcc"], 4),
                "recall_at_fpr_0.001": round(result["recall_at_fpr_0.001"], 4),
                "recall_at_fpr_0.01": round(result["recall_at_fpr_0.01"], 4),
            }
        )
    wall = (time.perf_counter() - started) / 60
    cpu = (time.process_time() - cpu_started) / 60
    # Wall time far above CPU time means sleep or throttling, not a slow fit.
    log.info(
        "variant %s finished in %.1f min wall, %.1f min CPU (ratio %.2f)",
        name, wall, cpu, cpu / max(wall, 1e-9),
    )
    if wall > 5 and cpu / max(wall, 1e-9) < 0.25:
        log.warning(
            "%s spent most of its wall time not computing - check that the "
            "machine did not sleep (powercfg /change standby-timeout-ac 0)", name,
        )
    return rows


def main() -> int:
    setup_logging(logfile=OUT_ROOT / "ablation.log")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    merged = load_config().path("interim") / "merged.parquet"
    if not merged.exists():
        log.error("%s not found. Run 'python -m frds prepare' first.", merged)
        return 1

    rows: list[dict] = []
    for name, overrides in VARIANTS:
        rows.extend(run_variant(name, overrides))
        # Written after every variant, so a killed run still leaves its results.
        pd.DataFrame(rows).to_csv(OUT_ROOT / "results.csv", index=False)

    frame = pd.DataFrame(rows)
    baseline = frame[frame["variant"] == "delay_0"].set_index("model")["auc_pr"]
    frame["auc_pr_retained_pct"] = [
        round(100 * row.auc_pr / baseline[row.model], 1) for row in frame.itertuples()
    ]
    frame.to_csv(OUT_ROOT / "results.csv", index=False)
    write_json(frame.to_dict(orient="records"), OUT_ROOT / "results.json")

    log.info("\n%s", frame.to_string(index=False))
    log.info("wrote %s", OUT_ROOT / "results.csv")
    plot(frame, load_config().path("reports") / "figures" / "label_delay_ablation.png")
    return 0


def plot(frame: pd.DataFrame, path: Path) -> None:
    """AUC-PR against confirmation delay, with the feature-removed floor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["delay_0", "delay_30", "delay_60"]
    days = [0, 30, 60]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for model, style in (("xgboost", "-o"), ("lightgbm", "--s")):
        part = frame[frame["model"] == model].set_index("variant")
        ax.plot(days, [part.loc[v, "auc_pr"] for v in order], style, label=model)
        if "none" in part.index:
            ax.axhline(part.loc["none", "auc_pr"], linestyle=":", alpha=0.6,
                       label=f"{model}, fraud-history features removed")
    ax.set_xlabel("Days before a fraud outcome is known")
    ax.set_ylabel("Test AUC-PR")
    ax.set_title("Sensitivity of performance to label-confirmation delay")
    ax.set_xticks(days)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", path.name)


if __name__ == "__main__":
    raise SystemExit(main())

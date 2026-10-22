"""Does re-expressing time-growing features actually help?

    python scripts/experiment_stationarity.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from frds import pipeline                                      # noqa: E402
from frds.config import load_config                            # noqa: E402
from frds.evaluation import metrics                            # noqa: E402
from frds.models.train import load_scores                      # noqa: E402
from frds.utils import get_logger, setup_logging, write_json   # noqa: E402

log = get_logger("frds.stationarity")

OUT_ROOT = REPO_ROOT / "reports" / "experiment_stationarity"
MODELS = ["xgboost", "lightgbm"]

CELLS: list[tuple[str, dict]] = [
    ("baseline_delay_0", {"features.stationarise": False, "features.label_delay_days": 0}),
    ("stationary_delay_0", {"features.stationarise": True, "features.label_delay_days": 0}),
    ("baseline_delay_30", {"features.stationarise": False, "features.label_delay_days": 30}),
    ("stationary_delay_30", {"features.stationarise": True, "features.label_delay_days": 30}),
]


def run_cell(name: str, overrides: dict) -> list[dict]:
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

    log.info("=== cell %s: %s ===", name, overrides)
    wall_start, cpu_start = time.perf_counter(), time.process_time()
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
        r = metrics.evaluate(y_test, scores[key], fpr_targets, label=model)
        rows.append({
            "cell": name,
            "stationarise": bool(overrides["features.stationarise"]),
            "label_delay_days": overrides["features.label_delay_days"],
            "model": model,
            "auc_pr": round(r["auc_pr"], 4),
            "auc_roc": round(r["auc_roc"], 4),
            "f1_best": round(r["f1_best"], 4),
            "mcc": round(r["mcc"], 4),
            "recall_at_fpr_0.001": round(r["recall_at_fpr_0.001"], 4),
        })

    wall = (time.perf_counter() - wall_start) / 60
    cpu = (time.process_time() - cpu_start) / 60
    log.info("cell %s finished in %.1f min wall, %.1f min CPU", name, wall, cpu)
    if wall > 5 and cpu / max(wall, 1e-9) < 0.25:
        log.warning("%s spent most of its wall time not computing - did the machine sleep?", name)
    return rows


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging(logfile=OUT_ROOT / "experiment.log")

    if not (load_config().path("interim") / "merged.parquet").exists():
        log.error("merged.parquet not found. Run 'python -m frds prepare' first.")
        return 1

    rows: list[dict] = []
    for name, overrides in CELLS:
        rows.extend(run_cell(name, overrides))
        pd.DataFrame(rows).to_csv(OUT_ROOT / "results.csv", index=False)

    frame = pd.DataFrame(rows)
    if frame.empty:
        log.error("no cell completed")
        return 1

    # Effect of stationarising, held at each delay setting.
    deltas = []
    for delay in sorted(frame.label_delay_days.unique()):
        for model in MODELS:
            sub = frame[(frame.label_delay_days == delay) & (frame.model == model)]
            base = sub[~sub.stationarise]
            stat = sub[sub.stationarise]
            if base.empty or stat.empty:
                continue
            b, s = float(base.auc_pr.iloc[0]), float(stat.auc_pr.iloc[0])
            deltas.append({
                "label_delay_days": delay, "model": model,
                "baseline_auc_pr": b, "stationary_auc_pr": s,
                "delta": round(s - b, 4),
                "relative_change_pct": round(100 * (s - b) / b, 2) if b else None,
            })
    delta_frame = pd.DataFrame(deltas)
    delta_frame.to_csv(OUT_ROOT / "effect.csv", index=False)
    write_json(delta_frame.to_dict(orient="records"), OUT_ROOT / "effect.json")

    log.info("\n%s", frame.to_string(index=False))
    log.info("\nEffect of stationarising:\n%s", delta_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

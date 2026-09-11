"""Do the behavioural improvements earn their place, under a realistic delay?

Three changes are tested here, added one at a time so that each can be judged
on its own rather than as part of an undifferentiated bundle:

    aggregates  per-entity running means and this row's ratio to them, plus two
                alternative account definitions frequency-encoded
    recency     exponentially down-weight older training rows
    refit       after early stopping fixes the iteration count, refit on
                train + validation before scoring test

All three are behavioural rather than label-derived, which is the reason they
are worth testing: the report's headline figure depends heavily on fraud
outcomes being confirmed instantly, and improvements that only show up in that
setting improve a number already argued to be optimistic. So the ladder is run
at a 30-day label-confirmation delay, which is the setting that corresponds to
something deployable, and the final configuration is also run without delay so
it can be placed in the main comparison table.

    python scripts/experiment_feature_package.py
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

log = get_logger("frds.package")

OUT_ROOT = REPO_ROOT / "reports" / "experiment_feature_package"
MODELS = ["xgboost", "lightgbm"]

#: Set from the outcome of scripts/experiment_stationarity.py: at a 30-day
#: delay it was worth +0.043 AUC-PR to XGBoost and +0.004 to LightGBM, and
#: was neutral without delay. Kept on - it never hurt materially, it lowers
#: measured drift, and it brings the two models into closer agreement.
STATIONARISE = True

BASE = {"features.stationarise": STATIONARISE}
AGG = {"features.entity_aggregates": True}
REC = {"imbalance.recency_half_life_days": 45}
REFIT = {"split.refit_on_valid": True}


def cell(name: str, delay: int, *layers: dict) -> tuple[str, dict]:
    overrides = {"features.label_delay_days": delay, **BASE}
    for layer in layers:
        overrides.update(layer)
    return name, overrides


CELLS: list[tuple[str, dict]] = [
    cell("d30_current", 30),
    cell("d30_aggregates", 30, AGG),
    cell("d30_agg_recency", 30, AGG, REC),
    cell("d30_agg_recency_refit", 30, AGG, REC, REFIT),
    cell("d0_full", 0, AGG, REC, REFIT),
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
            "label_delay_days": overrides["features.label_delay_days"],
            "aggregates": bool(overrides.get("features.entity_aggregates", False)),
            "recency": overrides.get("imbalance.recency_half_life_days") or 0,
            "refit": bool(overrides.get("split.refit_on_valid", False)),
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

    # Marginal contribution of each rung of the ladder, at the 30-day delay.
    ladder = ["d30_current", "d30_aggregates", "d30_agg_recency", "d30_agg_recency_refit"]
    steps = []
    for model in MODELS:
        sub = frame[frame.model == model].set_index("cell")
        for prev, curr in zip(ladder, ladder[1:]):
            if prev not in sub.index or curr not in sub.index:
                continue
            before, after = float(sub.loc[prev, "auc_pr"]), float(sub.loc[curr, "auc_pr"])
            steps.append({
                "model": model, "from": prev, "to": curr,
                "auc_pr_before": before, "auc_pr_after": after,
                "delta": round(after - before, 4),
                "relative_change_pct": round(100 * (after - before) / before, 2) if before else None,
            })
    step_frame = pd.DataFrame(steps)
    step_frame.to_csv(OUT_ROOT / "marginal_effect.csv", index=False)
    write_json(step_frame.to_dict(orient="records"), OUT_ROOT / "marginal_effect.json")

    log.info("\n%s", frame.to_string(index=False))
    log.info("\nMarginal effect of each change at a 30-day delay:\n%s",
             step_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

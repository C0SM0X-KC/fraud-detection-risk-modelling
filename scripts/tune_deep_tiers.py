"""Hyper-parameter search for the sequential and relational tiers.

    python scripts/tune_deep_tiers.py
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from frds.config import load_config                            # noqa: E402
from frds.data.split import load_split                         # noqa: E402
from frds.evaluation import metrics                            # noqa: E402
from frds.features.build import load_features                  # noqa: E402
from frds.models import imbalance, registry                    # noqa: E402
from frds.utils import get_logger, setup_logging, write_json   # noqa: E402

log = get_logger("frds.tune")

OUT_ROOT = REPO_ROOT / "reports" / "tuning"

#: Small candidate grids: enough to tell whether these tiers can work at all.
GRIDS: dict[str, dict[str, list]] = {
    "lstm": {
        "entity": ["card", "uid"],
        "window": [12, 24],
        "hidden": [64, 128],
        "epochs": [25],
        "lr": [0.001],
        "dropout": [0.2],
        "batch_size": [1024],
    },
    "hetero_gnn": {
        "hidden": [64, 128],
        "memory_decay": [0.5, 0.8, 0.95],
        "lr": [0.001, 0.002],
        "epochs": [20],
        "dropout": [0.2],
        "batch_size": [2048],
    },
}


def expand(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def fit_and_score(name: str, params: dict, ctx_bundle: dict) -> dict:
    X = ctx_bundle["X"]
    y = ctx_bundle["y"]
    split = ctx_bundle["split"]
    cfg = ctx_bundle["cfg"]

    scorer = registry.build(name, cfg)
    scorer.params.update(params)
    # The scorer reads its own params in __init__, so rebuild the derived
    # attributes after overriding them.
    for key, value in params.items():
        if hasattr(scorer, key):
            setattr(scorer, key, value)

    fit_ctx = imbalance.make_context(
        y.to_numpy()[split.train], cfg, ctx_bundle["categorical"],
        X, ctx_bundle["entity_keys"], ctx_bundle["times"],
    )

    started = time.perf_counter()
    scorer.fit(
        X.iloc[split.train], y.to_numpy()[split.train], fit_ctx,
        X.iloc[split.valid], y.to_numpy()[split.valid],
    )
    valid_scores = scorer.predict_proba(X.iloc[split.valid])
    test_scores = scorer.predict_proba(X.iloc[split.test])

    valid_ap = metrics.evaluate(y.to_numpy()[split.valid], valid_scores, label=name)["auc_pr"]
    test_ap = metrics.evaluate(y.to_numpy()[split.test], test_scores, label=name)["auc_pr"]

    row = {
        "model": name,
        **params,
        "valid_auc_pr": round(valid_ap, 4),
        "test_auc_pr": round(test_ap, 4),
        "fit_seconds": round(time.perf_counter() - started, 1),
    }
    log.info(
        "%s %s -> valid AUC-PR %.4f (test %.4f) in %.0fs",
        name, params, valid_ap, test_ap, row["fit_seconds"],
    )
    return row


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    setup_logging(logfile=OUT_ROOT / "tuning.log")

    cfg = load_config()
    X, y, categorical, keys = load_features(cfg)
    split = load_split(cfg)
    bundle = {
        "X": X,
        "y": y,
        "split": split,
        "cfg": cfg,
        "categorical": categorical,
        "entity_keys": {k: v for k, v in keys.items() if not k.startswith("__")},
        "times": keys["__time"],
    }

    rows: list[dict] = []
    for name, grid in GRIDS.items():
        candidates = expand(grid)
        log.info("=== %s: %d candidates ===", name, len(candidates))
        for i, params in enumerate(candidates, start=1):
            log.info("%s candidate %d/%d", name, i, len(candidates))
            try:
                rows.append(fit_and_score(name, params, bundle))
            except Exception as exc:
                log.error("%s %s failed: %s", name, params, exc)
            # Written after every candidate so a killed run keeps its progress.
            pd.DataFrame(rows).to_csv(OUT_ROOT / "deep_tier_search.csv", index=False)

    frame = pd.DataFrame(rows)
    if frame.empty:
        log.error("no candidate completed")
        return 1

    best = (
        frame.sort_values("valid_auc_pr", ascending=False)
        .groupby("model", as_index=False)
        .first()
    )
    write_json(best.to_dict(orient="records"), OUT_ROOT / "best_config.json")

    log.info("\n%s", frame.sort_values(["model", "valid_auc_pr"], ascending=[True, False]).to_string(index=False))
    log.info("\nBest per model (selected on validation):\n%s", best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

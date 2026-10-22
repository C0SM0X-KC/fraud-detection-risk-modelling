"""Adopt or reject the account profile and CatBoost - on validation only.

    python scripts/compare_feature_sets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "baseline_handset"


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(scores, span):
    z = (logit(scores["xgboost"][span]) + logit(scores["lightgbm"][span])) / 2
    return 1 / (1 + np.exp(-z))


def checkpoints(directory: Path) -> dict:
    """Per-model score checkpoints. The baseline is rebuilt from these rather
    than from a combined base_scores.npz, which a later run may overwrite."""
    return {m: np.load(directory / f"{m}.npz") for m in ("xgboost", "lightgbm")}


def decision_row(path: Path) -> dict:
    frame = pd.read_csv(path)
    rows = frame[frame["label"].str.startswith("decision layer")]
    return rows.iloc[0].to_dict() if len(rows) else {}


def main() -> int:
    combined = np.load(ROOT / "artifacts" / "base_scores.npz", allow_pickle=True)
    y_valid, y_test = combined["y_valid"], combined["y_test"]
    base_scores = checkpoints(BASE / "artifacts" / "scores")
    new_scores = checkpoints(ROOT / "artifacts" / "scores")
    for scores in (base_scores, new_scores):
        if len(scores["xgboost"]["valid"]) != len(y_valid):
            print("validation rows differ between runs; the comparison would not be like for like")
            return 1
    base_fusion = json.loads((BASE / "fusion_summary.json").read_text())
    new_fusion = json.loads((ROOT / "reports" / "metrics" / "fusion_summary.json").read_text())
    base_spec = json.loads((BASE / "artifacts" / "feature_spec.json").read_text())
    new_spec = json.loads((ROOT / "data" / "processed" / "feature_spec.json").read_text())

    profile_only = {
        "base_valid": float(average_precision_score(y_valid, blend(base_scores, "valid"))),
        "profile_valid": float(average_precision_score(y_valid, blend(new_scores, "valid"))),
        "base_test": float(average_precision_score(y_test, blend(base_scores, "test"))),
        "profile_test": float(average_precision_score(y_test, blend(new_scores, "test"))),
    }

    base_sel, new_sel = base_fusion["selected_score"], new_fusion["selected_score"]
    result = {
        "rule": "adopt if the decision layer's selected score has higher validation AUC-PR",
        "base_features": base_spec["n_features"],
        "profile_features": new_spec["n_features"],
        "base_selected": base_sel,
        "profile_selected": new_sel,
        "profile_members": new_fusion.get("blend_members"),
        "base_validation": base_fusion["validation_ranking"][base_sel],
        "profile_validation": new_fusion["validation_ranking"][new_sel],
        "base_test": decision_row(BASE / "model_comparison.csv").get("auc_pr"),
        "profile_test": decision_row(ROOT / "reports" / "tables" / "model_comparison.csv").get("auc_pr"),
        "account_profile_alone_xgb_lgbm_blend": profile_only,
        "catboost_validation": new_fusion["validation_ranking"].get("catboost"),
    }
    result["adopted"] = bool(result["profile_validation"] > result["base_validation"])
    result["note"] = (
        "" if result["adopted"] else
        "The account-profile run and its models are kept under reports/profile_run/ "
        "for reference."
    )
    out = ROOT / "reports" / "metrics" / "feature_adoption.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

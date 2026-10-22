"""Transfer check: the same protocol on Bank Account Fraud (report Table 3.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import Config
from .evaluation import metrics
from .fusion.calibration import ProbabilityCalibrator, expected_calibration_error
from .fusion.risk import policy_from_config, risk_score
from .fusion.stacking import MeanLogitBlend
from .models import registry
from .models.base import FitContext, positive_class_weight
from .utils import get_logger, set_seed, write_json

log = get_logger("frds.transfer")

CATEGORICAL = ("payment_type", "employment_status", "housing_status", "source", "device_os")


def load_baf(cfg: Config) -> pd.DataFrame:
    path = cfg.path("raw") / cfg.require("transfer.baf_file")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download Base.csv from "
            "kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 "
            "into data/raw/baf/."
        )
    frame = pd.read_csv(path)
    log.info("BAF %s, fraud rate %.3f%%", frame.shape,
             100 * frame[cfg.require("transfer.target_col")].mean())
    return frame


def prepare(frame: pd.DataFrame, cfg: Config):
    target = cfg.require("transfer.target_col")
    month = cfg.require("transfer.month_col")
    spans = {k: frame[month].isin(cfg.require(f"transfer.{k}_months")).to_numpy()
             for k in ("train", "valid", "test")}

    X = frame.drop(columns=[target, month]).copy()
    categorical = []
    for col in CATEGORICAL:
        if col in X:
            # Category order from the training months only; unseen -> -1.
            levels = sorted(X.loc[spans["train"], col].astype(str).unique())
            X[col] = pd.Categorical(X[col].astype(str), categories=levels).codes.astype(np.int32)
            categorical.append(col)
    return X, frame[target].to_numpy().astype(int), spans, categorical


def fairness(y: np.ndarray, scores: np.ndarray, age: np.ndarray, age_cut: int,
             target_fpr: float) -> dict:
    """FPR of each age group at the threshold giving ``target_fpr`` overall."""
    negatives = scores[y == 0]
    threshold = float(np.quantile(negatives, 1 - target_fpr))
    flagged = scores > threshold
    out = {"threshold": threshold, "overall_fpr": float(flagged[y == 0].mean()),
           "recall": float(flagged[y == 1].mean())}
    for name, mask in (("under", age < age_cut), ("over", age >= age_cut)):
        neg = (y == 0) & mask
        out[f"fpr_age_{name}_{age_cut}"] = float(flagged[neg].mean()) if neg.any() else np.nan
    a, b = out[f"fpr_age_under_{age_cut}"], out[f"fpr_age_over_{age_cut}"]
    out["fpr_ratio_min_over_max"] = float(min(a, b) / max(a, b)) if max(a, b) > 0 else np.nan
    return out


def run(cfg: Config) -> pd.DataFrame:
    seed = int(cfg.require("project.seed"))
    set_seed(seed)
    frame = load_baf(cfg)
    X, y, spans, categorical = prepare(frame, cfg)
    tr, va, te = spans["train"], spans["valid"], spans["test"]
    fpr_targets = sorted(set(list(cfg.require("evaluation.fpr_targets")) + [0.05]))

    valid_scores, test_scores = {}, {}
    for name in cfg.require("transfer.models"):
        scorer = registry.build(name, cfg)
        ctx = FitContext(categorical=categorical, seed=seed,
                         class_weight=positive_class_weight(y[tr]))
        log.info("BAF: fitting %s", name)
        scorer.fit(X[tr], y[tr], ctx, X[va], y[va])
        valid_scores[name] = scorer.predict_proba(X[va])
        test_scores[name] = scorer.predict_proba(X[te])

    # Same decision-layer rule as IEEE-CIS: the best candidate on validation.
    candidates = dict(valid_scores)
    tabular = [n for n in valid_scores if n in ("xgboost", "lightgbm")]
    if len(tabular) == 2:
        blend = MeanLogitBlend(tabular)
        candidates["blend_top2"] = blend.predict_proba(np.column_stack([valid_scores[n] for n in tabular]))
        test_scores["blend_top2"] = blend.predict_proba(np.column_stack([test_scores[n] for n in tabular]))
    ranking = {n: float(average_precision_score(y[va], s)) for n, s in candidates.items()}
    selected = max(ranking, key=ranking.get)
    log.info("BAF validation AUC-PR: %s -> %s",
             {k: round(v, 4) for k, v in ranking.items()}, selected)

    calibrator = ProbabilityCalibrator("isotonic").fit(candidates[selected], y[va])
    p_test = calibrator.transform(test_scores[selected])

    rows = [metrics.evaluate(y[te], s, fpr_targets, label=n) for n, s in test_scores.items()]
    rows.append(metrics.evaluate(y[te], p_test, fpr_targets,
                                 label=f"decision layer ({selected}; calibrated)"))
    # Built from the full metric rows: the shared comparison table keeps a
    # fixed IEEE-CIS column set, and recall at 5 % FPR is BAF's own headline.
    keep = ["label", "auc_pr", "auc_pr_lift_over_prevalence", "auc_roc", "f1_best",
            "recall_at_fpr_0.01", "recall_at_fpr_0.05"]
    table = (pd.DataFrame(rows)[keep].sort_values("auc_pr", ascending=False)
             .reset_index(drop=True))

    policy = policy_from_config(cfg)
    tiers = policy.assign(risk_score(p_test, cfg))
    tier_rows = []
    for tier in ("low", "medium", "high"):
        mask = tiers == tier
        tier_rows.append({
            "tier": tier, "applications": int(mask.sum()),
            "share_of_volume": float(mask.mean()),
            "fraud_caught": int(y[te][mask].sum()),
            "share_of_fraud": float(y[te][mask].sum() / max(y[te].sum(), 1)),
            "precision": float(y[te][mask].mean()) if mask.any() else float("nan"),
        })

    age_col = "customer_age"
    fair = fairness(y[te], p_test, X.loc[te, age_col].to_numpy(),
                    int(cfg.get("transfer.fairness_age_threshold", 50)),
                    float(cfg.get("transfer.fairness_fpr", 0.05))) if age_col in X else {}

    reports = cfg.path("reports")
    table.to_csv(reports / "tables" / "baf_transfer.csv", index=False)
    pd.DataFrame(tier_rows).to_csv(reports / "tables" / "baf_tiers.csv", index=False)
    write_json({
        "rows": {k: int(v.sum()) for k, v in spans.items()},
        "prevalence": {k: float(y[v].mean()) for k, v in spans.items()},
        "validation_ranking": ranking,
        "selected": selected,
        "ece_after_calibration": expected_calibration_error(y[te], p_test),
        "model_comparison": table.to_dict(orient="records"),
        "tiers": tier_rows,
        "fairness": fair,
    }, reports / "metrics" / "baf_transfer.json")
    log.info("\n%s", table.to_string(index=False))
    if fair:
        log.info("fairness at %.0f%% FPR: FPR ratio (min/max across age groups) %.3f",
                 100 * float(cfg.get("transfer.fairness_fpr", 0.05)),
                 fair["fpr_ratio_min_over_max"])
    return table

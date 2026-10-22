"""Bayesian hyper-parameter search for the tabular tier (report Table 3.2, B.2)."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ..config import Config
from ..data.split import load_split
from ..features.build import load_features
from ..utils import get_logger, set_seed
from .base import positive_class_weight
from .train import split_validation_span

log = get_logger("frds.models.tuning")


def _space(trial, name: str) -> dict:
    if name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 16, 512, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        }
    raise ValueError(f"no search space for '{name}' (tunable: xgboost, lightgbm)")


def _fit_and_score(name, params, X_fit, y_fit, X_eval, y_eval, categorical,
                   seed) -> tuple[float, int]:
    weight = positive_class_weight(y_fit)
    if name == "xgboost":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
            n_estimators=3000, early_stopping_rounds=100, scale_pos_weight=weight,
            random_state=seed, n_jobs=-1, **params,
        )
        model.fit(X_fit, y_fit, eval_set=[(X_eval, y_eval)], verbose=False)
        p = model.predict_proba(X_eval)[:, 1]
        best = int(model.best_iteration)
    else:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            objective="binary", metric="average_precision", n_estimators=3000,
            subsample_freq=1, scale_pos_weight=weight, random_state=seed,
            n_jobs=-1, verbose=-1, **params,
        )
        model.fit(
            X_fit, y_fit, eval_set=[(X_eval, y_eval)], eval_metric="average_precision",
            categorical_feature=[c for c in categorical if c in X_fit.columns] or "auto",
            callbacks=[lgb.early_stopping(100, verbose=False, first_metric_only=True)],
        )
        p = model.predict_proba(X_eval)[:, 1]
        best = int(model.best_iteration_ or 0)
    return float(average_precision_score(y_eval, p)), best


def tune(cfg: Config, names: list[str], n_trials: int | None = None,
         timeout: int | None = None) -> dict:
    import optuna

    seed = int(cfg.require("project.seed"))
    set_seed(seed)
    n_trials = int(n_trials or cfg.get("tuning.n_trials", 30))
    timeout = int(timeout or cfg.get("tuning.timeout_seconds", 3600))
    neg_frac = float(cfg.get("tuning.negative_sample_frac", 0.3))

    X, y, categorical, keys = load_features(cfg)
    split = load_split(cfg)
    y_arr = y.to_numpy()
    valid_refit, _ = split_validation_span(split.valid, keys["__time"], cfg, y_arr)
    eval_idx = valid_refit if valid_refit.size else split.valid

    rng = np.random.default_rng(seed)
    train = split.train
    negatives = train[y_arr[train] == 0]
    kept = np.sort(np.concatenate([
        train[y_arr[train] == 1],
        rng.choice(negatives, int(len(negatives) * neg_frac), replace=False),
    ]))
    X_fit, y_fit = X.iloc[kept], y_arr[kept]
    X_eval, y_eval = X.iloc[eval_idx], y_arr[eval_idx]
    log.info("tuning on %d training rows (%d fraud, negatives at %.0f%%), scored on %d "
             "validation rows", len(kept), int(y_fit.sum()), neg_frac * 100, len(eval_idx))

    path = cfg.path("artifacts") / "tuned_params.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    tables = cfg.path("reports") / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    for name in names:
        baseline_params = {k: v for k, v in (cfg.get(f"models.{name}", {}) or {}).items()
                           if k not in ("n_estimators", "early_stopping_rounds", "tree_method",
                                        "n_jobs", "subsample_freq")}
        started = time.perf_counter()
        baseline, _ = _fit_and_score(name, baseline_params, X_fit, y_fit, X_eval, y_eval,
                                     categorical, seed)
        log.info("%s hand-set parameters: validation AUC-PR %.4f", name, baseline)

        def objective(trial):
            params = _space(trial, name)
            score, best = _fit_and_score(name, params, X_fit, y_fit, X_eval, y_eval,
                                         categorical, seed)
            trial.set_user_attr("best_iteration", best)
            return score

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            study_name=f"frds-{name}",
        )
        # Start from the hand-set configuration, so the search can only
        # improve on what the report already uses.
        study.enqueue_trial({k: v for k, v in baseline_params.items()
                             if k in _defaults(name)})
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def report(study, trial) -> None:
            log.info("%s trial %d: validation AUC-PR %.4f (best so far %.4f)",
                     name, trial.number, trial.value or float("nan"), study.best_value)

        study.optimize(objective, n_trials=n_trials, timeout=timeout,
                       show_progress_bar=False, callbacks=[report])

        history = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs",
                                                "duration", "state"))
        history = history.sort_values("value", ascending=False)
        history.to_csv(tables / f"tuning_{name}.csv", index=False)

        best = study.best_trial
        results[name] = {
            "params": best.params,
            "validation_auc_pr": round(best.value, 5),
            "hand_set_validation_auc_pr": round(baseline, 5),
            "best_iteration": best.user_attrs.get("best_iteration"),
            "trials": len(study.trials),
            "seconds": round(time.perf_counter() - started, 1),
            "training_negative_fraction": neg_frac,
        }
        log.info("%s best validation AUC-PR %.4f (hand-set %.4f) after %d trials",
                 name, best.value, baseline, len(study.trials))

    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote %s; set tuning.use_tuned=true to train with these", path.name)
    return results


def _defaults(name: str) -> dict:
    """The search space's keys, with the report's hand-set values."""
    if name == "xgboost":
        return {"max_depth": 8, "learning_rate": 0.03, "subsample": 0.8,
                "colsample_bytree": 0.6, "min_child_weight": 5.0, "reg_lambda": 1.0,
                "reg_alpha": 0.01, "gamma": 0.0}
    return {"num_leaves": 128, "learning_rate": 0.03, "min_child_samples": 50,
            "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 1.0,
            "reg_alpha": 0.01, "min_split_gain": 0.0}

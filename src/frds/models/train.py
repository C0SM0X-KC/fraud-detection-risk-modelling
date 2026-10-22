"""Model layer orchestration (L3)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..data.split import SplitIndex, expanding_window_folds
from ..utils import get_logger, timed, write_json
from . import imbalance, registry
from .base import BaseScorer, positive_class_weight

log = get_logger("frds.models.train")


@dataclass
class ScoreSet:
    """Base-model scores for one model across the three spans."""

    name: str
    oof: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    fit_seconds: float
    best_iteration: int | None = None


def _fit_one(
    scorer: BaseScorer,
    X: pd.DataFrame,
    y: np.ndarray,
    fit_idx: np.ndarray,
    cfg: Config,
    categorical: list[str],
    eval_idx: np.ndarray | None = None,
    entity_keys: dict[str, np.ndarray] | None = None,
    times: np.ndarray | None = None,
) -> BaseScorer:
    X_fit, y_fit = X.iloc[fit_idx], y[fit_idx]
    fit_times = times[fit_idx] if times is not None else None
    X_fit, y_fit = imbalance.apply(X_fit, y_fit, cfg)
    if fit_times is not None and len(fit_times) != len(y_fit):
        # Resampling changed the row count, so per-row ages no longer align.
        fit_times = None
    # The sequence and graph tiers need the whole time-ordered frame: a row's
    # history and its graph neighbours lie outside the fold being fitted.
    ctx = imbalance.make_context(
        y_fit, cfg, categorical, X, entity_keys, times, fit_times
    )

    X_eval = y_eval = None
    if eval_idx is not None and len(eval_idx):
        X_eval, y_eval = X.iloc[eval_idx], y[eval_idx]

    scorer.fit(X_fit, y_fit, ctx, X_eval, y_eval)
    return scorer



def split_validation_span(
    valid_idx: np.ndarray, time_values: np.ndarray, cfg: Config,
    y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut the validation span into (refit part, calibration holdout)."""
    if not bool(cfg.get("split.refit_on_valid", False)):
        return np.empty(0, dtype=valid_idx.dtype), valid_idx

    frac = float(cfg.get("split.calibration_holdout_frac", 0.3))
    if not 0.0 < frac < 1.0:
        raise ValueError(
            f"split.calibration_holdout_frac must be strictly between 0 and 1, got {frac}"
        )

    t = time_values[valid_idx]
    cut = float(np.quantile(t, 1.0 - frac))
    refit_part = valid_idx[t <= cut]
    holdout = valid_idx[t > cut]
    if holdout.size == 0 or refit_part.size == 0:
        log.warning("validation span too short to cut; using it all for calibration")
        return np.empty(0, dtype=valid_idx.dtype), valid_idx

    # Too few positives to calibrate reliably: keep the whole span for calibration, skip the refit.
    minimum = int(cfg.get("split.min_calibration_positives", 200))
    if y is not None:
        positives = int(np.sum(y[holdout]))
        if positives < minimum:
            log.warning(
                "calibration holdout has only %d positives (floor %d); using the "
                "whole validation span for calibration and skipping the refit",
                positives, minimum,
            )
            return np.empty(0, dtype=valid_idx.dtype), valid_idx

    log.info(
        "validation span split: %d rows join the refit, %d held out for "
        "calibration and score selection",
        len(refit_part), len(holdout),
    )
    return refit_part, holdout


def _fingerprint(
    X: pd.DataFrame, split: SplitIndex, cfg: Config, n_folds: int
) -> str:
    """Identity of the run a checkpoint belongs to."""
    payload = {
        "rows": int(len(X)),
        "columns": list(X.columns),
        "train": int(len(split.train)),
        "valid": int(len(split.valid)),
        "test": int(len(split.test)),
        "n_folds": int(n_folds),
        "imbalance": str(cfg.get("imbalance.method", "class_weight")),
        # Protocol settings that change stored scores, so they invalidate a checkpoint.
        "refit_on_valid": bool(cfg.get("split.refit_on_valid", False)),
        "calibration_holdout_frac": float(cfg.get("split.calibration_holdout_frac", 0.0)),
        "recency_half_life_days": cfg.get("imbalance.recency_half_life_days"),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _checkpoint_path(cfg: Config, name: str) -> Path:
    directory = cfg.path("artifacts") / "scores"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}.npz"


def save_checkpoint(res: "ScoreSet", cfg: Config, fingerprint: str, params: dict) -> Path:
    path = _checkpoint_path(cfg, res.name)
    np.savez_compressed(
        path,
        oof=res.oof,
        valid=res.valid,
        test=res.test,
        fit_seconds=np.array(res.fit_seconds),
        best_iteration=np.array(-1 if res.best_iteration is None else res.best_iteration),
        fingerprint=np.array(fingerprint),
        params=np.array(json.dumps(params, sort_keys=True, default=str)),
    )
    log.info("checkpointed %s -> %s", res.name, path.name)
    return path


def load_checkpoint(
    cfg: Config, name: str, fingerprint: str, params: dict
) -> "ScoreSet | None":
    path = _checkpoint_path(cfg, name)
    if not path.exists():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        stored_fingerprint = str(z["fingerprint"])
        stored_params = str(z["params"])
    except Exception as exc:  # a truncated file from a killed process
        log.warning("ignoring unreadable checkpoint %s: %s", path.name, exc)
        return None

    if stored_fingerprint != fingerprint:
        log.info("%s checkpoint is stale (data or split changed); refitting", name)
        return None
    if stored_params != json.dumps(params, sort_keys=True, default=str):
        log.info("%s hyper-parameters changed since the checkpoint; refitting", name)
        return None

    best = int(z["best_iteration"])
    log.info("reusing checkpoint for %s (fitted in %.1fs)", name, float(z["fit_seconds"]))
    return ScoreSet(
        name=name,
        oof=z["oof"],
        valid=z["valid"],
        test=z["test"],
        fit_seconds=float(z["fit_seconds"]),
        best_iteration=None if best < 0 else best,
    )



def _refit_including_valid(
    name: str,
    fitted: BaseScorer,
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_refit: np.ndarray,
    valid_cal: np.ndarray,
    test_idx: np.ndarray,
    cfg: Config,
    categorical: list[str],
    entity_keys: dict[str, np.ndarray] | None,
    time_values: np.ndarray,
    fallback_valid: np.ndarray,
    fallback_test: np.ndarray,
) -> tuple[BaseScorer, np.ndarray, np.ndarray]:
    """Refit on train + the earlier part of validation, then re-score."""
    best_iter = getattr(fitted.model, "best_iteration", None)
    if best_iter is None:
        best_iter = getattr(fitted.model, "best_iteration_", None)

    refit = registry.build(name, cfg)
    if isinstance(best_iter, (int, np.integer)) and best_iter > 0:
        refit.params["n_estimators"] = int(best_iter)
        refit.params.pop("early_stopping_rounds", None)
        log.info("%s refitting on train + validation head for %d iterations",
                 name, int(best_iter))
    else:
        log.info("%s refitting on train + validation head", name)

    combined = np.concatenate([train_idx, valid_refit])
    try:
        _fit_one(refit, X, y, combined, cfg, categorical, None,
                 entity_keys, time_values)
        return (
            refit,
            refit.predict_proba(X.iloc[valid_cal]),
            refit.predict_proba(X.iloc[test_idx]),
        )
    except Exception as exc:
        # A tier that cannot refit keeps the train-only model rather than
        # failing the whole benchmark.
        log.error("%s refit failed (%s); keeping the train-only fit", name, exc)
        return fitted, fallback_valid, fallback_test



def train_base_models(
    X: pd.DataFrame,
    y: pd.Series,
    split: SplitIndex,
    cfg: Config,
    categorical: list[str],
    time_values: np.ndarray,
    entity_keys: dict[str, np.ndarray] | None = None,
    force_refit: bool = False,
) -> tuple[dict[str, ScoreSet], np.ndarray]:
    """Fit every enabled scorer and return its three score vectors."""
    y_arr = y.to_numpy()
    names = list(cfg.require("models.enabled"))
    n_folds = int(cfg.require("split.n_oof_folds"))

    train_times = time_values[split.train]
    folds = list(expanding_window_folds(train_times, n_folds))
    oof_local = np.concatenate([score_idx for _, score_idx in folds])
    oof_positions = split.train[oof_local]
    log.info(
        "out-of-fold coverage: %d of %d train rows across %d expanding folds",
        len(oof_positions), len(split.train), len(folds),
    )

    results: dict[str, ScoreSet] = {}
    model_dir = cfg.path("artifacts") / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(X, split, cfg, n_folds)
    log.info("run fingerprint %s", fingerprint)

    valid_refit, valid_cal = split_validation_span(
        split.valid, time_values, cfg, y_arr
    )

    for name in names:
        params = registry.model_params(name, cfg)
        if not force_refit:
            cached = load_checkpoint(cfg, name, fingerprint, params)
            if cached is not None:
                results[name] = cached
                continue

        oof_scores = np.full(len(oof_local), np.nan, dtype=np.float64)
        cursor = 0
        started = time.perf_counter()

        try:
            for k, (fit_local, score_local) in enumerate(folds, start=1):
                fit_idx = split.train[fit_local]
                score_idx = split.train[score_local]
                # The last 5% of each fit window is the early-stopping set.
                holdout = max(1, int(len(fit_idx) * 0.05))
                inner_fit, inner_eval = fit_idx[:-holdout], fit_idx[-holdout:]

                scorer = registry.build(name, cfg)
                _fit_one(scorer, X, y_arr, inner_fit, cfg, categorical, inner_eval,
                         entity_keys, time_values)
                preds = scorer.predict_proba(X.iloc[score_idx])
                oof_scores[cursor: cursor + len(score_idx)] = preds
                cursor += len(score_idx)
                log.info("  %s fold %d/%d scored %d rows", name, k, len(folds),
                         len(score_idx))

            # Final fit on the train span. Early stopping uses the part of
            # validation that may later join the refit, never the holdout.
            early_stop_idx = valid_refit if valid_refit.size else valid_cal
            final = registry.build(name, cfg)
            _fit_one(final, X, y_arr, split.train, cfg, categorical, early_stop_idx,
                     entity_keys, time_values)
            valid_scores = final.predict_proba(X.iloc[valid_cal])
            test_scores = final.predict_proba(X.iloc[split.test])

            if valid_refit.size:
                final, valid_scores, test_scores = _refit_including_valid(
                    name, final, X, y_arr, split.train, valid_refit, valid_cal,
                    split.test, cfg, categorical, entity_keys, time_values,
                    valid_scores, test_scores,
                )
            final.save(model_dir)
        except ImportError as exc:
            # An optional tier whose dependency is absent is skipped with a
            # message rather than taking the whole benchmark down with it.
            log.error("skipping %s: %s", name, exc)
            continue

        best_iter = getattr(final.model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(final.model, "best_iteration_", None)
        if not isinstance(best_iter, (int, np.integer)):
            best_iter = None

        results[name] = ScoreSet(
            name=name,
            oof=oof_scores,
            valid=valid_scores,
            test=test_scores,
            fit_seconds=time.perf_counter() - started,
            best_iteration=int(best_iter) if best_iter else None,
        )
        log.info("%s complete in %.1fs", name, results[name].fit_seconds)
        # Written immediately: a run that dies three models in should not cost
        # the three that already succeeded.
        save_checkpoint(results[name], cfg, fingerprint, params)

        importance = final.feature_importance()
        if importance is not None:
            importance.head(50).to_csv(
                cfg.path("reports") / "tables" / f"importance_{name}.csv",
                header=["importance"],
            )

    return results, oof_positions, valid_cal


def save_scores(
    results: dict[str, ScoreSet],
    oof_positions: np.ndarray,
    split: SplitIndex,
    y: pd.Series,
    cfg: Config,
    valid_positions: np.ndarray | None = None,
) -> Path:
    path = cfg.path("artifacts") / "base_scores.npz"
    if valid_positions is None:
        valid_positions = split.valid
    payload: dict[str, np.ndarray] = {
        "oof_positions": oof_positions,
        "valid_positions": valid_positions,
        "test_positions": split.test,
        "y_oof": y.to_numpy()[oof_positions],
        "y_valid": y.to_numpy()[valid_positions],
        "y_test": y.to_numpy()[split.test],
        "model_names": np.array(list(results), dtype=object),
    }
    for name, res in results.items():
        payload[f"oof__{name}"] = res.oof
        payload[f"valid__{name}"] = res.valid
        payload[f"test__{name}"] = res.test

    np.savez_compressed(path, **payload)
    write_json(
        {
            name: {
                "fit_seconds": round(res.fit_seconds, 2),
                "best_iteration": res.best_iteration,
            }
            for name, res in results.items()
        },
        cfg.path("reports") / "metrics" / "base_model_fits.json",
    )
    log.info("wrote %s", path.name)
    return path


def load_scores(cfg: Config) -> dict[str, np.ndarray]:
    path = cfg.path("artifacts") / "base_scores.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m frds train' first.")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def score_matrix(scores: dict[str, np.ndarray], span: str) -> tuple[np.ndarray, list[str]]:
    """Stack per-model score vectors into the meta-learner's design matrix."""
    names = [str(n) for n in scores["model_names"]]
    cols = [scores[f"{span}__{n}"] for n in names]
    return np.column_stack(cols), names

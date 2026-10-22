"""Imbalance handling, applied to the training fold only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger
from .base import FitContext, positive_class_weight

log = get_logger("frds.models.imbalance")


SECONDS_PER_DAY = 86_400


def recency_weights(fit_times: np.ndarray | None, cfg: Config) -> np.ndarray | None:
    """Exponentially down-weight older training rows."""
    half_life = cfg.get("imbalance.recency_half_life_days")
    if not half_life or fit_times is None or len(fit_times) == 0:
        return None

    age_days = (float(np.max(fit_times)) - np.asarray(fit_times, dtype=np.float64)) / SECONDS_PER_DAY
    weights = np.power(0.5, age_days / float(half_life)).astype(np.float64)
    log.info(
        "recency weighting: half-life %.0f days, weights %.4f - %.4f",
        float(half_life), float(weights.min()), float(weights.max()),
    )
    return weights


def make_context(
    y_train: np.ndarray,
    cfg: Config,
    categorical: list[str],
    full_X: pd.DataFrame | None = None,
    entity_keys: dict[str, np.ndarray] | None = None,
    times: np.ndarray | None = None,
    fit_times: np.ndarray | None = None,
) -> FitContext:
    return FitContext(
        categorical=list(categorical),
        seed=int(cfg.require("project.seed")),
        class_weight=positive_class_weight(y_train),
        sample_weight=recency_weights(fit_times, cfg),
        full_X=full_X,
        entity_keys=entity_keys,
        times=times,
    )


def apply(
    X_train: pd.DataFrame, y_train: np.ndarray, cfg: Config
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a possibly resampled training fold."""
    method = str(cfg.get("imbalance.method", "class_weight")).lower()
    if method in {"class_weight", "none"}:
        return X_train, y_train

    ratio = float(cfg.get("imbalance.sampling_strategy", 0.1))
    seed = int(cfg.require("project.seed"))

    # SMOTE and ADASYN interpolate between neighbours, so they cannot see NaN.
    filled = X_train.fillna(X_train.median(numeric_only=True)).fillna(0.0)

    if method == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(sampling_strategy=ratio, random_state=seed, k_neighbors=5)
    elif method == "adasyn":
        from imblearn.over_sampling import ADASYN

        sampler = ADASYN(sampling_strategy=ratio, random_state=seed, n_neighbors=5)
    else:
        raise ValueError(f"unknown imbalance.method: {method}")

    before = int(y_train.sum())
    X_res, y_res = sampler.fit_resample(filled, y_train)
    log.info(
        "%s: %d -> %d rows, positives %d -> %d",
        method, len(y_train), len(y_res), before, int(np.sum(y_res)),
    )
    return pd.DataFrame(X_res, columns=X_train.columns), np.asarray(y_res)

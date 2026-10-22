"""Decision layer (L4), part 1: the stacking meta-learner."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from ..utils import get_logger

log = get_logger("frds.fusion.stacking")

EPS = 1e-6


def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


class StackingMetaLearner:
    """Fuse base-model probabilities into one score."""

    def __init__(self, seed: int = 42, C: float = 1.0) -> None:
        self.seed = seed
        self.C = C
        self.model: LogisticRegression | None = None
        self.model_names: list[str] = []

    def fit(self, base_scores: np.ndarray, y: np.ndarray, model_names: list[str]) -> "StackingMetaLearner":
        self.model_names = list(model_names)
        Z = to_logit(base_scores)
        self.model = LogisticRegression(
            C=self.C, max_iter=2000, class_weight="balanced", random_state=self.seed
        )
        self.model.fit(Z, y)
        weights = dict(zip(self.model_names, np.round(self.model.coef_.ravel(), 4)))
        log.info("meta-learner weights (logit space): %s", weights)
        return self

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("meta-learner is not fitted")
        return self.model.predict_proba(to_logit(base_scores))[:, 1]

    def weights(self) -> dict[str, float]:
        if self.model is None:
            return {}
        return {
            name: float(w)
            for name, w in zip(self.model_names, self.model.coef_.ravel())
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "names": self.model_names}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "StackingMetaLearner":
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload["model"]
        obj.model_names = payload["names"]
        return obj


class MeanLogitBlend:
    """Average of base-model log-odds: fusion with nothing to fit."""

    def __init__(self, model_names: list[str] | None = None) -> None:
        self.model_names = list(model_names or [])

    def predict_proba(self, base_scores: np.ndarray) -> np.ndarray:
        z = to_logit(base_scores).mean(axis=1)
        return 1.0 / (1.0 + np.exp(-z))

    def weights(self) -> dict[str, float]:
        return {n: 1.0 / len(self.model_names) for n in self.model_names}

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"kind": "mean_logit_blend", "names": self.model_names}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "MeanLogitBlend":
        return cls(joblib.load(path)["names"])


def load_combiner(artifacts: Path, selected: str):
    """The fitted fusion object behind a selected score, or None for a single model."""
    if selected == "stacked":
        return StackingMetaLearner.load(Path(artifacts) / "meta_learner.joblib")
    if selected.startswith("blend"):
        return MeanLogitBlend.load(Path(artifacts) / "blend.joblib")
    return None

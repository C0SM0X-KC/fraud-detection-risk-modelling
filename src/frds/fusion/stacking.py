"""Decision layer (L4), part 1: the stacking meta-learner.

Report section 4.5, fusion tier. A logistic regression over the base-model
probabilities, fitted on the out-of-fold scores from the train span. Logit
transformation first, because the base scores are bunched near zero and a
linear model on raw probabilities cannot separate 0.001 from 0.02 - which is
exactly the range the decision depends on.
"""

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

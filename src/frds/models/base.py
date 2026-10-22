"""Common contract for every scorer in the model layer (L3)."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("frds.models")


@dataclass
class FitContext:
    """Everything a model needs beyond X and y."""

    categorical: list[str] = field(default_factory=list)
    seed: int = 42
    class_weight: float = 1.0          # scale_pos_weight equivalent
    sample_weight: np.ndarray | None = None
    full_X: "pd.DataFrame | None" = None
    entity_keys: dict[str, np.ndarray] | None = None
    times: np.ndarray | None = None


class BaseScorer(abc.ABC):
    """A fitted-once, scored-many fraud probability model."""

    name: str = "base"
    #: models that cannot see NaN get an imputer wrapped around them
    handles_missing: bool = False
    #: models whose training cost makes tuning or full-data fits impractical
    supports_categorical: bool = False

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})
        self.model: Any = None
        self.feature_names_: list[str] = []

    @abc.abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        ctx: FitContext,
        X_valid: pd.DataFrame | None = None,
        y_valid: np.ndarray | None = None,
    ) -> "BaseScorer":
        ...

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fraud) as a 1-D float array aligned to X's rows."""

    # -- persistence ----------------------------------------------------
    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.joblib"
        joblib.dump({"params": self.params, "model": self.model,
                     "features": self.feature_names_}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "BaseScorer":
        payload = joblib.load(path)
        if "model" not in payload:
            # Torch tiers are refitted, not reloaded (their context needs the feature frame).
            raise NotImplementedError(
                f"{cls.name} is not reloadable from disk in this build "
                f"({path.name} holds a state_dict, not a fitted estimator)."
            )
        obj = cls(payload["params"])
        obj.model = payload["model"]
        obj.feature_names_ = payload["features"]
        return obj

    # -- helpers --------------------------------------------------------
    def _remember_columns(self, X: pd.DataFrame) -> None:
        self.feature_names_ = list(X.columns)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.feature_names_ and list(X.columns) != self.feature_names_:
            missing = set(self.feature_names_) - set(X.columns)
            if missing:
                raise ValueError(f"{self.name}: missing feature columns {sorted(missing)[:5]}")
            X = X[self.feature_names_]
        return X

    def feature_importance(self) -> pd.Series | None:
        """Native importance where the model exposes one; SHAP is used for the
        per-alert explanations, this is only for a global sanity read."""
        model = self.model
        est = getattr(model, "named_steps", {}).get("estimator", model) if hasattr(model, "named_steps") else model
        if hasattr(est, "feature_importances_"):
            return pd.Series(est.feature_importances_, index=self.feature_names_).sort_values(
                ascending=False
            )
        if hasattr(est, "coef_"):
            return pd.Series(np.abs(est.coef_.ravel()), index=self.feature_names_).sort_values(
                ascending=False
            )
        return None


def positive_class_weight(y: np.ndarray) -> float:
    """scale_pos_weight = negatives / positives, the algorithm-level answer to
    imbalance the report prefers over resampling (Table 4.1, step A.5)."""
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return negatives / positives

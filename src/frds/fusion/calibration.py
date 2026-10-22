"""Decision layer (L4), part 2: probability calibration."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..utils import get_logger

log = get_logger("frds.fusion.calibration")


class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic") -> None:
        self.method = method.lower()
        self.model = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        p_raw = np.asarray(p_raw, dtype=np.float64)
        if self.method == "isotonic":
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.model.fit(p_raw, y)
        elif self.method in {"platt", "sigmoid"}:
            self.model = LogisticRegression(max_iter=1000)
            self.model.fit(p_raw.reshape(-1, 1), y)
        elif self.method == "none":
            self.model = None
        else:
            raise ValueError(f"unknown calibration method: {self.method}")

        log.info(
            "calibrated on %d rows (%d positive) with %s",
            len(y), int(np.sum(y)), self.method,
        )
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype=np.float64)
        if self.method == "none" or self.model is None:
            return p_raw
        if self.method == "isotonic":
            return np.clip(self.model.predict(p_raw), 0.0, 1.0)
        return self.model.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"method": self.method, "model": self.model}, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ProbabilityCalibrator":
        payload = joblib.load(path)
        obj = cls(payload["method"])
        obj.model = payload["model"]
        return obj


def reliability_curve(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> dict[str, list[float]]:
    """Empirical fraud rate per predicted-probability bin."""
    p = np.asarray(p, dtype=np.float64)
    y_true = np.asarray(y_true)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(p.min(), p.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    predicted, observed, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        predicted.append(float(p[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))

    return {"predicted": predicted, "observed": observed, "count": counts}


def expected_calibration_error(
    y_true: np.ndarray, p: np.ndarray, n_bins: int = 15
) -> float:
    curve = reliability_curve(y_true, p, n_bins)
    total = sum(curve["count"])
    if total == 0:
        return float("nan")
    return float(
        sum(
            c * abs(pred - obs)
            for pred, obs, c in zip(curve["predicted"], curve["observed"], curve["count"])
        )
        / total
    )

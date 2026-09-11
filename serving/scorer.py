"""The serving path: features in, risk decision out.

This is the deployed half of the decision layer. It loads exactly what the
fusion stage chose and nothing more - which, when that stage selects a single
model over the stack, means the demo loads one model rather than six. That is
not a shortcut: it is what the evaluated system actually decided to use, and
serving the five tiers it rejected would misrepresent it.

The one asymmetry worth knowing is the graph tier. Its scores depend on entity
memories that accumulate as traffic flows past, so it cannot be loaded cold and
asked about a single transaction. If the decision layer selects it (or selects
the stack, which contains it), the memories are warmed to the start of the
replay window first, and scoring then advances them online - which is how a
live deployment would run it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from frds.config import Config, load_config
from frds.data.split import load_split
from frds.features.build import load_features
from frds.fusion.calibration import ProbabilityCalibrator
from frds.fusion.risk import TierPolicy, decide, policy_from_config
from frds.fusion.stacking import StackingMetaLearner
from frds.models.graph import HeteroGNNScorer
from frds.models.registry import build
from frds.models.sequence import SequenceScorer
from frds.utils import get_logger, read_json

log = get_logger("frds.serving")


@dataclass
class Decision:
    """One scored transaction, as an analyst queue would hold it."""

    position: int
    transaction_id: int
    amount: float
    day: float
    risk_score: int
    probability: float
    tier: str
    action: str
    is_fraud: int | None = None


class FraudScorer:
    """Loads the decision layer and scores batches of transactions."""

    def __init__(self, cfg: Config | None = None, warm_to: int | None = None) -> None:
        self.cfg = cfg or load_config()
        self.X, self.y, self.categorical, self.keys = load_features(self.cfg)
        self.split = load_split(self.cfg)
        self.times = self.keys["__time"]
        self.amounts = self.keys["__amount"]
        self.entity_keys = {k: v for k, v in self.keys.items() if not k.startswith("__")}

        artifacts = self.cfg.path("artifacts")
        summary = read_json(self.cfg.path("reports") / "metrics" / "fusion_summary.json")
        self.selected: str = summary.get("selected_score", "stacked")
        self.calibration_method: str = summary.get("calibration_method", "isotonic")

        self.calibrator = ProbabilityCalibrator.load(artifacts / "calibrator.joblib")
        self.meta: StackingMetaLearner | None = None
        self.model_names: list[str] = []
        self.models: list = []

        if self.selected == "stacked":
            self.meta = StackingMetaLearner.load(artifacts / "meta_learner.joblib")
            self.model_names = list(self.meta.model_names)
        else:
            self.model_names = [self.selected]

        self._load_models(warm_to)
        self.policy: TierPolicy = policy_from_config(self.cfg)
        log.info(
            "serving '%s' (%s calibration), tiers medium>=%d high>=%d",
            self.selected, self.calibration_method,
            self.policy.medium_lo, self.policy.high_lo,
        )

    # -- loading --------------------------------------------------------
    def _load_models(self, warm_to: int | None) -> None:
        model_dir = self.cfg.path("artifacts") / "models"
        for name in self.model_names:
            path = model_dir / f"{name}.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run 'python -m frds train' first."
                )
            scorer = type(build(name, self.cfg)).load(path)
            if isinstance(scorer, (SequenceScorer, HeteroGNNScorer)):
                scorer.restore_context(self.X, self.entity_keys)
            if isinstance(scorer, HeteroGNNScorer):
                target = warm_to if warm_to is not None else int(self.split.test.min())
                log.info("warming %s entity memories to row %d", name, target)
                scorer.warm_memory(target)
            self.models.append(scorer)
            log.info("loaded %s", name)

    # -- scoring --------------------------------------------------------
    def base_scores(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for scorer in self.models:
            if isinstance(scorer, HeteroGNNScorer):
                columns.append(scorer.predict_online(frame))
            else:
                columns.append(scorer.predict_proba(frame))
        return np.column_stack(columns)

    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        base = self.base_scores(frame)
        raw = self.meta.predict_proba(base) if self.meta is not None else base[:, 0]
        return self.calibrator.transform(raw)

    def score_positions(self, positions: np.ndarray) -> pd.DataFrame:
        """Score rows by their position in the feature frame."""
        positions = np.asarray(positions)
        frame = self.X.iloc[positions]
        p = self.probability(frame)

        decisions = decide(
            p, self.amounts[positions], self.times[positions], self.cfg, self.policy
        )
        decisions.insert(0, "position", positions)
        decisions.insert(1, "day", np.round(self.times[positions] / 86_400, 2))
        decisions.insert(2, "amount", np.round(self.amounts[positions], 2))
        # The outcome is shown in the demo so an analyst view can be judged
        # against the truth. A live system would not have it at decision time.
        decisions["is_fraud"] = self.y.to_numpy()[positions]
        return decisions

    # -- explanation ----------------------------------------------------
    def explain_positions(self, positions: np.ndarray, top_k: int = 5) -> pd.DataFrame:
        """SHAP reason codes for specific rows, from the strongest tree model."""
        from frds.evaluation.explain import reason_codes, tree_shap_values

        tree = next(
            (m for m in self.models
             if not isinstance(m, (SequenceScorer, HeteroGNNScorer))
             and hasattr(getattr(m, "model", None), "feature_importances_")),
            None,
        )
        if tree is None:
            return pd.DataFrame()

        subset = self.X.iloc[np.asarray(positions)].reset_index(drop=True)
        values = tree_shap_values(tree.model, subset)
        frame = reason_codes(values, subset, np.arange(len(subset)), top_k)
        frame["position"] = np.asarray(positions)
        return frame

    # -- helpers --------------------------------------------------------
    def test_positions(self) -> np.ndarray:
        return self.split.test

    def describe(self) -> dict:
        return {
            "selected_score": self.selected,
            "models_loaded": self.model_names,
            "calibration": self.calibration_method,
            "tier_boundaries": {
                "medium": self.policy.medium_lo,
                "high": self.policy.high_lo,
            },
            "features": int(self.X.shape[1]),
            "test_rows": int(len(self.split.test)),
        }

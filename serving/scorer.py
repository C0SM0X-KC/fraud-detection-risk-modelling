"""The serving path: features in, risk decision out."""

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
from frds.fusion.stacking import load_combiner
from frds.product import FastTree, make_explainer
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
        self.models: list = []
        self.meta = load_combiner(artifacts, self.selected)
        self.model_names = (list(self.meta.model_names) if self.meta is not None
                            else [self.selected])

        self._load_models(warm_to)
        self.policy: TierPolicy = policy_from_config(self.cfg)

        # Native single-row prediction and TreeSHAP for the tree tiers; the
        # sequence and graph tiers keep their own scoring paths.
        self.fast = [
            None if isinstance(m, (SequenceScorer, HeteroGNNScorer)) else FastTree(m)
            for m in self.models
        ]
        trees = [f for f in self.fast if f is not None]
        self.explainer = (make_explainer(trees, self.meta)[0]
                          if len(trees) == len(self.fast) else
                          next((f for f in trees if f.explainable), None))
        state = read_json(self.cfg.path("processed") / "feature_spec.json")
        self.categories = None
        try:
            from frds.features.online import load_state

            fitted = load_state(self.cfg)["fitted"]
            self.categories = {c: v["categories"]
                               for c, v in fitted.get("categoricals", {}).items()}
        except (FileNotFoundError, RuntimeError):
            pass
        self.n_features = int(state.get("n_features", self.X.shape[1]))
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
        for scorer, fast in zip(self.models, self.fast):
            if isinstance(scorer, HeteroGNNScorer):
                columns.append(scorer.predict_online(frame))
            elif fast is not None:
                columns.append(fast.predict(frame))
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
    def explain_position(self, position: int, top_k: int = 5) -> dict | None:
        """Plain-language reason codes for one row, from the deciding tree model."""
        from frds.evaluation.reasons import explain_row

        if self.explainer is None:
            return None
        frame = self.X.iloc[[int(position)]]
        values, base = self.explainer.contributions(frame)
        out = explain_row(values[0], frame.iloc[0], None, self.categories, top_k)
        out["base_value_log_odds"] = round(float(base[0]), 4)
        out["explained_model"] = self.explainer.scorer.name
        return out

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
            "features": self.n_features,
            "test_rows": int(len(self.split.test)),
        }

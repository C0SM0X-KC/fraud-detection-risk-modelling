"""The product: one transaction in, a decision with its reasons out."""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .config import Config
from .evaluation.reasons import explain_row
from .features.online import OnlineFeatureStore, load_state
from .fusion.calibration import ProbabilityCalibrator
from .fusion.risk import per_transaction_threshold, policy_from_config, risk_score
from .fusion.stacking import MeanLogitBlend, load_combiner
from .models.registry import build
from .utils import get_logger, read_json

log = get_logger("frds.product")

VERDICTS = {
    "low": ("Legitimate", "Approve"),
    "medium": ("Suspicious", "Step-up authentication"),
    "high": ("Likely fraud", "Hold for manual review"),
}

#: Tiers whose models need a history frame rather than one feature row; they
#: cannot score a transaction the online store has just featurised.
_FRAME_BOUND = {"lstm", "tcn", "hetero_gnn"}


class FastTree:
    """Native single-row prediction and TreeSHAP for a boosted tree model."""

    def __init__(self, scorer) -> None:
        self.scorer = scorer
        self.kind = "generic"
        model = scorer.model
        try:
            import xgboost as xgb

            if isinstance(model, xgb.XGBClassifier):
                self.kind = "xgboost"
                self._xgb = xgb
                self.booster = model.get_booster()
                try:
                    self.iteration_range = (0, int(model.best_iteration) + 1)
                except AttributeError:
                    self.iteration_range = (0, 0)
        except ImportError:
            pass
        if self.kind == "generic" and type(model).__name__ == "CatBoostClassifier":
            self.kind = "catboost"
        if self.kind == "generic":
            try:
                import lightgbm as lgb

                if isinstance(model, lgb.LGBMClassifier):
                    self.kind = "lightgbm"
                    self.booster = model.booster_
                    best = getattr(model, "best_iteration_", None)
                    self.num_iteration = int(best) if best else None
            except ImportError:
                pass

    @property
    def explainable(self) -> bool:
        return self.kind in {"xgboost", "lightgbm", "catboost"}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.kind == "xgboost":
            arr = frame.to_numpy(np.float32)
            return np.asarray(self.booster.inplace_predict(
                arr, iteration_range=self.iteration_range, validate_features=False,
            ), dtype=np.float64)
        if self.kind == "lightgbm":
            return np.asarray(self.booster.predict(
                frame.to_numpy(np.float64), num_iteration=self.num_iteration,
            ), dtype=np.float64)
        return self.scorer.predict_proba(frame)

    def contributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(SHAP values per feature, base value) in log-odds, per row."""
        if self.kind == "xgboost":
            matrix = self._xgb.DMatrix(frame.to_numpy(np.float32), missing=np.nan)
            out = self.booster.predict(
                matrix, pred_contribs=True, iteration_range=self.iteration_range,
                validate_features=False,
            )
        elif self.kind == "lightgbm":
            out = self.booster.predict(
                frame.to_numpy(np.float64), pred_contrib=True,
                num_iteration=self.num_iteration,
            )
        elif self.kind == "catboost":
            from catboost import Pool

            prepared = self.scorer._prepare(frame)
            out = self.scorer.model.get_feature_importance(
                Pool(prepared, cat_features=self.scorer.categorical_), type="ShapValues")
        else:
            raise NotImplementedError(f"no native attribution for {self.scorer.name}")
        out = np.asarray(out, dtype=np.float64)
        return out[:, :-1], out[:, -1]


class BlendExplainer:
    """Exact TreeSHAP for an equal-weight logit blend of tree models."""

    def __init__(self, members: list[FastTree]) -> None:
        self.members = members

        class _Name:
            name = "blend(" + "+".join(m.scorer.name for m in members) + ")"
        self.scorer = _Name()

    explainable = True

    def contributions(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        parts = [m.contributions(frame) for m in self.members]
        values = np.mean([v for v, _ in parts], axis=0)
        base = np.mean([b for _, b in parts], axis=0)
        return values, base


def make_explainer(models: list[FastTree], combiner) -> tuple[object | None, bool]:
    """(explainer, whether it explains the deciding score exactly)."""
    if combiner is None:
        m = models[0]
        return (m, True) if m.explainable else (None, False)
    if isinstance(combiner, MeanLogitBlend) and all(m.explainable for m in models):
        return BlendExplainer(models), True
    proxy = next((m for m in models if m.explainable), None)
    return proxy, False


class DecisionService:
    """Online featurisation, scoring, calibration, tiering and explanation."""

    def __init__(self, cfg: Config, store: OnlineFeatureStore | None = None) -> None:
        self.cfg = cfg
        artifacts = cfg.path("artifacts")
        summary = read_json(cfg.path("reports") / "metrics" / "fusion_summary.json")
        self.selected: str = summary.get("selected_score", "stacked")
        self.calibration_method: str = summary.get("calibration_method", "isotonic")
        self.calibrator = ProbabilityCalibrator.load(artifacts / "calibrator.joblib")
        self.policy = policy_from_config(cfg)

        self.meta = load_combiner(artifacts, self.selected)
        names = list(self.meta.model_names) if self.meta is not None else [self.selected]
        bound = sorted(set(names) & _FRAME_BOUND)
        if bound:
            raise NotImplementedError(
                f"the decision layer uses {bound}, which score from a history frame "
                "rather than a feature row; serve a tabular selection for online use"
            )

        self.models: list[FastTree] = []
        for name in names:
            scorer = type(build(name, cfg)).load(artifacts / "models" / f"{name}.joblib")
            self.models.append(FastTree(scorer))

        # Reasons come from the deciding score (or a learned stack's strongest tree member).
        self.explainer, self.explains_decision = make_explainer(self.models, self.meta)

        state = None
        if store is None:
            state = load_state(cfg)
            store = OnlineFeatureStore(state, cfg)
        self.store = store
        fitted = (state or store.state)["fitted"]
        self.categories = {c: s["categories"] for c, s in fitted.get("categoricals", {}).items()}
        log.info("decision service ready: '%s' (%s calibration), %d history rows",
                 self.selected, self.calibration_method, self.store.history_rows)

    # -- core ---------------------------------------------------------------
    def _probability(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        base = np.column_stack([m.predict(features) for m in self.models])
        raw = self.meta.predict_proba(base) if self.meta is not None else base[:, 0]
        return raw, self.calibrator.transform(raw)

    def _decide(self, p: np.ndarray, amounts: np.ndarray) -> dict[str, np.ndarray]:
        scores = risk_score(p, self.cfg)
        tiers = self.policy.assign(scores)
        break_even = per_transaction_threshold(amounts, self.cfg)
        return {"scores": scores, "tiers": tiers, "break_even": break_even,
                "cost_flag": p > break_even}

    def _decision_record(self, txn_id, row, p_raw, p, decided, i) -> dict:
        tier = str(decided["tiers"][i])
        verdict, action = VERDICTS[tier]
        return {
            "transaction_id": int(txn_id),
            "risk_score": int(decided["scores"][i]),
            "probability": round(float(p[i]), 6),
            "raw_model_score": round(float(p_raw[i]), 6),
            "tier": tier,
            "verdict": verdict,
            "action": action,
            "policy_action": self.policy.action_for(tier),
            "is_fraud_prediction": bool(decided["cost_flag"][i]),
            "break_even_probability": round(float(decided["break_even"][i]), 6),
            "amount": row.amount,
            "model": self.selected,
        }

    # -- public -------------------------------------------------------------
    def score(
        self,
        raw: Mapping[str, Any],
        remember: bool = True,
        explain: bool = True,
        top_k: int = 5,
    ) -> dict:
        """Score one transaction end to end, with per-stage timings."""
        timings = {}
        t0 = time.perf_counter()
        row = self.store.frame(raw)
        codes = self.store._codes(row)
        history = {e: {"known": self.store._degree(e, codes[e]) > 0,
                       "prior_transactions": self.store._degree(e, codes[e])}
                   for e in ("card", "uid", "device", "email")}
        features = self.store._features(row, codes)
        t1 = time.perf_counter()
        p_raw, p = self._probability(features)
        t2 = time.perf_counter()
        decided = self._decide(p, np.array([row.amount]))
        txn_id = row.txn_id
        if remember:
            self.store._append(row, codes, label=None)
        record = self._decision_record(txn_id, row, p_raw, p, decided, 0)
        t3 = time.perf_counter()
        timings["features_ms"] = (t1 - t0) * 1000
        timings["model_ms"] = (t2 - t1) * 1000
        timings["decision_ms"] = (t3 - t2) * 1000

        if explain and self.explainer is not None:
            values, base = self.explainer.contributions(features)
            raw_dict = {k: v for k, v in dict(raw).items()}
            record["reasons"] = explain_row(values[0], features.iloc[0], raw_dict,
                                            self.categories, top_k)
            record["reasons"]["base_value_log_odds"] = round(float(base[0]), 4)
            record["reasons"]["explained_model"] = self.explainer.scorer.name
            record["reasons"]["explains_decision_model"] = self.explains_decision
            timings["explanation_ms"] = (time.perf_counter() - t3) * 1000
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        record["history"] = history
        record["timings_ms"] = {k: round(v, 3) for k, v in timings.items()}
        return record

    def score_many(
        self,
        raws: Iterable[Mapping[str, Any]],
        remember: bool = True,
        explain_tiers: Iterable[str] = (),
        top_k: int = 5,
    ) -> list[dict]:
        """Micro-batch: featurise in arrival order, score in one model call."""
        rows, frames, ids, raw_list = [], [], [], []
        for raw in raws:
            row = self.store.frame(raw)
            codes = self.store._codes(row)
            frames.append(self.store._features(row, codes))
            ids.append(row.txn_id)
            rows.append(row)
            raw_list.append(dict(raw))
            if remember:
                self.store._append(row, codes, label=None)
        if not frames:
            return []
        features = pd.concat(frames, ignore_index=True)
        p_raw, p = self._probability(features)
        amounts = np.array([r.amount for r in rows])
        decided = self._decide(p, amounts)
        out = [self._decision_record(ids[i], rows[i], p_raw, p, decided, i)
               for i in range(len(rows))]

        wanted = set(explain_tiers)
        targets = [i for i, d in enumerate(out) if d["tier"] in wanted]
        if targets and self.explainer is not None:
            values, base = self.explainer.contributions(features.iloc[targets])
            for k, i in enumerate(targets):
                reasons = explain_row(values[k], features.iloc[i], raw_list[i],
                                      self.categories, top_k)
                reasons["base_value_log_odds"] = round(float(base[k]), 4)
                reasons["explained_model"] = self.explainer.scorer.name
                out[i]["reasons"] = reasons
        return out

    def record_outcome(self, transaction_id: int, is_fraud: bool) -> bool:
        """Analyst feedback: the confirmed outcome joins the entity history."""
        return self.store.record_outcome(int(transaction_id), int(bool(is_fraud)))

    def templates(self) -> list[dict]:
        """Real transactions to start a check from (see online._templates)."""
        return self.store.state.get("templates", [])

    def describe(self) -> dict:
        return {
            "model": self.selected,
            "calibration": self.calibration_method,
            "tier_boundaries": {"medium": self.policy.medium_lo, "high": self.policy.high_lo},
            "history_rows": self.store.history_rows,
            "transactions_scored_online": len(self.store.arrivals),
            "explanations": None if self.explainer is None else self.explainer.scorer.name,
        }


class DeployedModel:
    """The selected score on precomputed feature rows: raw, calibrated, tiered."""

    def __init__(self, cfg: Config) -> None:
        artifacts = cfg.path("artifacts")
        summary = read_json(cfg.path("reports") / "metrics" / "fusion_summary.json")
        self.cfg = cfg
        self.selected = summary.get("selected_score", "stacked")
        self.meta = load_combiner(artifacts, self.selected)
        names = list(self.meta.model_names) if self.meta is not None else [self.selected]
        bound = sorted(set(names) & _FRAME_BOUND)
        if bound:
            raise NotImplementedError(f"the decision layer uses frame-bound tiers {bound}")
        self.models = [FastTree(type(build(n, cfg)).load(artifacts / "models" / f"{n}.joblib"))
                       for n in names]
        self.calibrator = ProbabilityCalibrator.load(artifacts / "calibrator.joblib")
        self.policy = policy_from_config(cfg)

    def raw(self, frame: pd.DataFrame) -> np.ndarray:
        base = np.column_stack([m.predict(frame) for m in self.models])
        return self.meta.predict_proba(base) if self.meta is not None else base[:, 0]

    def risk(self, frame: pd.DataFrame) -> np.ndarray:
        return risk_score(self.calibrator.transform(self.raw(frame)), self.cfg)

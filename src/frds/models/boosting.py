"""Tabular tier: XGBoost and LightGBM.

Both handle NaN natively, so no imputation is applied - the missingness
pattern in IEEE-CIS is itself informative and imputing it away costs signal.
Early stopping uses the chronological validation slice, never a random one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseScorer, FitContext, log


class XGBoostScorer(BaseScorer):
    name = "xgboost"
    handles_missing = True

    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        import xgboost as xgb

        self._remember_columns(X_train)
        params = dict(self.params)
        early = params.pop("early_stopping_rounds", 100)
        has_eval = X_valid is not None and y_valid is not None

        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=ctx.class_weight,
            random_state=ctx.seed,
            n_jobs=-1,
            early_stopping_rounds=early if has_eval else None,
            **params,
        )
        log.info("fitting %s on %s (scale_pos_weight=%.1f)", self.name, X_train.shape,
                 ctx.class_weight)
        self.model.fit(
            X_train,
            y_train,
            sample_weight=ctx.sample_weight,
            eval_set=[(X_valid, y_valid)] if has_eval else None,
            verbose=False,
        )
        if has_eval and getattr(self.model, "best_iteration", None) is not None:
            log.info("%s stopped at iteration %d", self.name, self.model.best_iteration)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._align(X))[:, 1]


class LightGBMScorer(BaseScorer):
    name = "lightgbm"
    handles_missing = True
    supports_categorical = True

    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        import lightgbm as lgb

        self._remember_columns(X_train)
        params = dict(self.params)
        early = params.pop("early_stopping_rounds", 100)
        has_eval = X_valid is not None and y_valid is not None

        self.model = lgb.LGBMClassifier(
            objective="binary",
            # Track average precision ONLY. LightGBM otherwise also tracks
            # binary_logloss, and with scale_pos_weight near 27 the weighted
            # logloss degrades within a handful of rounds while average
            # precision is still climbing. Early stopping then fires on the
            # wrong metric - in the first full run this stopped LightGBM at
            # iteration 10 of a possible 3000 and cost it 0.16 AUC-PR, which
            # would have been reported as "LightGBM is worse than XGBoost"
            # when it was really "LightGBM was stopped before it had started".
            metric="average_precision",
            scale_pos_weight=ctx.class_weight,
            random_state=ctx.seed,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
        callbacks = []
        if has_eval:
            callbacks = [
                lgb.early_stopping(early, verbose=False, first_metric_only=True),
                lgb.log_evaluation(period=0),
            ]

        # LightGBM can split on integer-coded categories directly, which is
        # cheaper and usually better than one-hot for high-cardinality fields.
        categorical = [c for c in ctx.categorical if c in X_train.columns]
        log.info("fitting %s on %s (%d categorical)", self.name, X_train.shape,
                 len(categorical))
        fit_kwargs: dict = {
            "sample_weight": ctx.sample_weight,
            "eval_metric": "average_precision",
            "categorical_feature": categorical or "auto",
            "callbacks": callbacks,
        }
        if has_eval:
            # LightGBM 4.7 deprecated eval_set in favour of eval_X / eval_y;
            # older releases only understand eval_set.
            import inspect

            if "eval_X" in inspect.signature(self.model.fit).parameters:
                fit_kwargs["eval_X"] = X_valid
                fit_kwargs["eval_y"] = y_valid
            else:
                fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        self.model.fit(X_train, y_train, **fit_kwargs)
        if has_eval and getattr(self.model, "best_iteration_", None):
            log.info("%s stopped at iteration %d", self.name, self.model.best_iteration_)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._align(X))[:, 1]

"""Tabular tier: XGBoost and LightGBM."""

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
            # Track average precision only: weighted logloss degrades early and stops training too soon.
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


class CatBoostScorer(BaseScorer):
    """Tabular tier, third member: CatBoost."""

    name = "catboost"
    handles_missing = True
    supports_categorical = True

    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        from catboost import CatBoostClassifier, Pool

        self._remember_columns(X_train)
        params = dict(self.params)
        early = params.pop("early_stopping_rounds", 100)
        iterations = params.pop("n_estimators", params.pop("iterations", 3000))
        has_eval = X_valid is not None and y_valid is not None
        categorical = [c for c in ctx.categorical if c in X_train.columns]
        self.categorical_ = categorical

        # Device stays out of self.params so GPU and CPU runs share checkpoints.
        import os

        device = os.environ.get("FRDS_CATBOOST_TASK_TYPE", "CPU").upper()
        self.model = CatBoostClassifier(
            iterations=iterations,
            task_type=device,
            loss_function="Logloss",
            eval_metric="PRAUC",
            scale_pos_weight=ctx.class_weight,
            random_seed=ctx.seed,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
            **params,
        )
        log.info("fitting %s on %s (%d categorical)", self.name, X_train.shape, len(categorical))
        train_pool = Pool(self._prepare(X_train), y_train, cat_features=categorical,
                          weight=ctx.sample_weight)
        eval_pool = (Pool(self._prepare(X_valid), y_valid, cat_features=categorical)
                     if has_eval else None)
        self.model.fit(train_pool, eval_set=eval_pool,
                       early_stopping_rounds=early if has_eval else None,
                       use_best_model=has_eval)
        if has_eval:
            log.info("%s stopped at iteration %d", self.name, self.model.get_best_iteration())
        return self

    @property
    def best_iteration_(self):
        return self.model.get_best_iteration() if self.model is not None else None

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._align(X)
        cats = [c for c in getattr(self, "categorical_", []) if c in X.columns]
        if cats:
            X = X.copy()
            X[cats] = X[cats].fillna(-1).astype(np.int64)
        return X

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._prepare(X))[:, 1]

    def save(self, directory):
        path = super().save(directory)
        import joblib

        payload = joblib.load(path)
        payload["categorical"] = getattr(self, "categorical_", [])
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path):
        import joblib

        obj = super().load(path)
        obj.categorical_ = joblib.load(path).get("categorical", [])
        return obj

"""Baseline tier: logistic regression and random forest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import BaseScorer, FitContext, log


class LogisticRegressionScorer(BaseScorer):
    name = "logistic_regression"
    handles_missing = False

    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        self._remember_columns(X_train)
        self.model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                (
                    "estimator",
                    LogisticRegression(
                        C=float(self.params.get("C", 0.1)),
                        max_iter=int(self.params.get("max_iter", 2000)),
                        class_weight="balanced",
                        solver="lbfgs",
                        n_jobs=None,
                        random_state=ctx.seed,
                    ),
                ),
            ]
        )
        log.info("fitting %s on %s", self.name, X_train.shape)
        self.model.fit(X_train, y_train, estimator__sample_weight=ctx.sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._align(X))[:, 1]


class RandomForestScorer(BaseScorer):
    name = "random_forest"
    handles_missing = False

    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        self._remember_columns(X_train)
        params: dict[str, Any] = {
            "n_estimators": 300,
            "max_depth": 18,
            "min_samples_leaf": 20,
            "n_jobs": -1,
            **self.params,
        }
        self.model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                (
                    "estimator",
                    RandomForestClassifier(
                        class_weight="balanced_subsample",
                        random_state=ctx.seed,
                        **params,
                    ),
                ),
            ]
        )
        log.info("fitting %s on %s", self.name, X_train.shape)
        self.model.fit(X_train, y_train, estimator__sample_weight=ctx.sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._align(X))[:, 1]

"""CatBoost, the third tabular member: fits, saves, reloads, and explains
itself exactly - the same guarantees the product path relies on for XGBoost
and LightGBM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from frds.evaluation.explain import tree_shap_values  # noqa: E402
from frds.models.base import FitContext  # noqa: E402
from frds.models.boosting import CatBoostScorer  # noqa: E402
from frds.product import FastTree  # noqa: E402


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 4_000
    X = pd.DataFrame({
        "amount": rng.gamma(2.0, 50.0, n),
        "velocity": rng.poisson(2, n).astype(float),
        "cat_device": rng.integers(-1, 6, n).astype(np.int32),
    })
    X.loc[rng.random(n) < 0.1, "amount"] = np.nan
    logit = 0.01 * X["amount"].fillna(50) + 0.8 * X["velocity"] + (X["cat_device"] == 3) * 1.5 - 5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int).to_numpy()
    ctx = FitContext(categorical=["cat_device"], seed=0, class_weight=3.0)
    scorer = CatBoostScorer({"n_estimators": 200, "depth": 4, "early_stopping_rounds": 20})
    scorer.fit(X.iloc[:3000], y[:3000], ctx, X.iloc[3000:], y[3000:])
    path = scorer.save(tmp_path_factory.mktemp("cb"))
    return scorer, CatBoostScorer.load(path), X


def test_reload_predicts_identically(fitted):
    scorer, reloaded, X = fitted
    np.testing.assert_allclose(scorer.predict_proba(X), reloaded.predict_proba(X))


def test_fast_path_and_shap_add_up(fitted):
    _, reloaded, X = fitted
    fast = FastTree(reloaded)
    assert fast.kind == "catboost" and fast.explainable
    rows = X.iloc[:200].astype(np.float64)        # the online store's float64 rows
    values, base = fast.contributions(rows)
    p = 1 / (1 + np.exp(-(values.sum(axis=1) + base)))
    np.testing.assert_allclose(p, fast.predict(rows), rtol=1e-5, atol=1e-6)


def test_evaluation_shap_matches_serving_shap(fitted):
    _, reloaded, X = fitted
    rows = X.iloc[:100]
    np.testing.assert_allclose(tree_shap_values(reloaded.model, rows),
                               FastTree(reloaded).contributions(rows)[0], rtol=1e-6, atol=1e-8)

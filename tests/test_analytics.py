"""Statistical helpers of the analytics stage, on data with a known answer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frds import analytics


def test_wilson_interval_contains_rate_and_stays_in_bounds():
    lo, hi = analytics.wilson_interval([0, 5, 100], [100, 100, 100])
    assert np.all(lo >= 0) and np.all(hi <= 1)
    assert lo[1] < 0.05 < hi[1]
    assert lo[0] == 0 and np.isclose(hi[2], 1)


def test_levels_pools_rare_values_and_keeps_missing():
    s = pd.Series(["a"] * 200 + ["b"] * 150 + ["c"] * 5 + [None] * 120)
    out = analytics.levels(s, top=1)
    assert set(out) == {"a", "other", "missing"}
    assert (out == "missing").sum() == 120


def test_tests_find_a_planted_signal_and_ignore_noise():
    rng = np.random.default_rng(0)
    n = 20_000
    y = (rng.random(n) < 0.1).astype(int)
    frame = pd.DataFrame({"isFraud": y})
    # One category and one number carry signal; the rest are independent noise.
    for col, _ in analytics.CATEGORICAL_TESTS:
        frame[col] = rng.choice(["x", "y"], n)
    frame["ProductCD"] = np.where(y == 1, rng.choice(["x", "y"], n, p=[0.8, 0.2]),
                                  rng.choice(["x", "y"], n))
    for col, _ in analytics.NUMERIC_TESTS:
        frame[col] = rng.normal(size=n)
    frame["C1"] = rng.normal(size=n) + y

    cat = analytics.categorical_tests(frame).set_index("column")
    assert cat.loc["ProductCD", "p_bonferroni"] < 1e-6
    assert cat.loc["ProductCD", "cramers_v"] > 0.1
    assert cat.drop("ProductCD")["cramers_v"].max() < 0.05

    num = analytics.numeric_tests(frame).set_index("column")
    assert num.loc["C1", "prob_fraud_higher"] > 0.7
    assert num.drop("C1")["rank_biserial"].abs().max() < 0.06


def test_bootstrap_brackets_the_estimate():
    rng = np.random.default_rng(1)
    n = 6_000
    y = (rng.random(n) < 0.1).astype(int)
    good = y + rng.normal(scale=0.5, size=n)
    weak = y + rng.normal(scale=2.0, size=n)
    tiers = np.where(good > 0.8, "high", np.where(good > 0.5, "medium", "low"))
    table, comparison = analytics.bootstrap_intervals(
        y, {"good": good, "weak": weak}, tiers, np.repeat(np.arange(30), n // 30),
        ("good", "weak"), n_boot=100, seed=0)
    assert ((table["ci_low"] <= table["estimate"]) & (table["estimate"] <= table["ci_high"])).all()
    assert comparison["ci_low"] > 0 and comparison["share_of_resamples_better"] == 1.0

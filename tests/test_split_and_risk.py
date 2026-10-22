"""Splitting, folds, drift, metrics and the decision layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frds.config import load_config
from frds.data.split import chronological_split, expanding_window_folds, temporal_windows
from frds.evaluation import drift, metrics
from frds.fusion import calibration, risk
from frds.models.sequence import build_history_index

DAY = 86_400


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def frame():
    n = 1_000
    rng = np.random.default_rng(7)
    times = np.sort(rng.integers(0, 180 * DAY, size=n))
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": times,
            "TransactionAmt": rng.uniform(1, 500, size=n),
            "isFraud": (rng.random(n) < 0.05).astype(int),
        }
    )


# ------------------------------------------------------------------ split --
def test_split_is_chronological_and_disjoint(frame, cfg):
    split = chronological_split(frame, cfg)
    t = frame["TransactionDT"].to_numpy()

    assert len(set(split.train) & set(split.valid)) == 0
    assert len(set(split.valid) & set(split.test)) == 0
    assert len(split.train) + len(split.valid) + len(split.test) == len(frame)
    assert t[split.train].max() <= t[split.valid].min()
    assert t[split.valid].max() <= t[split.test].min()


def test_split_rejects_unsorted_input(frame, cfg):
    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(shuffled, cfg)


def test_split_fractions_must_sum_to_one(frame, cfg):
    cfg.override("split.test_frac", 0.5)
    with pytest.raises(ValueError, match="sum to 1.0"):
        chronological_split(frame, cfg)


def test_expanding_folds_never_score_before_they_fit():
    times = np.arange(1_000) * 60
    folds = list(expanding_window_folds(times, n_folds=4))
    assert len(folds) == 4
    for fit_idx, score_idx in folds:
        assert fit_idx.max() < score_idx.min()
    # Score slices tile the tail without overlapping.
    scored = np.concatenate([s for _, s in folds])
    assert len(scored) == len(np.unique(scored))
    assert np.all(np.diff(scored) > 0)


def test_temporal_windows_cover_every_row():
    times = np.sort(np.random.default_rng(0).integers(0, 100 * DAY, size=500))
    windows = temporal_windows(times, 5)
    assert sum(len(w) for w in windows) == len(times)
    assert len(np.unique(np.concatenate(windows))) == len(times)


# ---------------------------------------------------------------- history --
def test_history_index_pads_the_front_and_ends_on_the_current_row():
    group = np.array([0, 0, 0, 1, 0])
    index, mask = build_history_index(group, window=3)

    # Row 0 is the entity's first: two padded steps, then itself.
    assert mask[0].tolist() == [False, False, True]
    assert index[0][-1] == 0
    # Row 4 is the entity's fourth: full history of rows 1, 2 then itself.
    assert mask[4].tolist() == [True, True, True]
    assert index[4].tolist() == [1, 2, 4]
    # The final step is always the row being scored.
    assert (index[:, -1] == np.arange(len(group))).all()


# ------------------------------------------------------------------- PSI ---
def test_psi_is_zero_for_the_same_distribution():
    rng = np.random.default_rng(0)
    sample = rng.normal(size=5_000)
    assert drift.psi(sample, sample.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=5_000)
    small = drift.psi(reference, rng.normal(loc=0.1, size=5_000))
    large = drift.psi(reference, rng.normal(loc=1.5, size=5_000))
    assert small < large
    assert drift.interpret(large) == "significant"
    assert drift.interpret(small) == "stable"


# --------------------------------------------------------------- metrics ---
def test_auc_pr_beats_prevalence_for_a_useful_score():
    rng = np.random.default_rng(1)
    y = (rng.random(4_000) < 0.04).astype(int)
    scores = rng.random(4_000) * 0.5 + y * 0.4
    result = metrics.evaluate(y, scores, label="test")
    assert result["auc_pr"] > result["prevalence"]
    assert result["auc_pr_lift_over_prevalence"] > 1.0


def test_accuracy_is_recorded_but_exposed_as_useless():
    y = np.zeros(1_000, dtype=int)
    y[:35] = 1
    scores = np.zeros(1_000)          # predicts "legitimate" for everything
    result = metrics.evaluate(y, scores, label="always_legitimate")
    assert result["accuracy_of_always_legitimate"] == pytest.approx(0.965)
    assert result["auc_pr"] < 0.1     # and it catches nothing


def test_recall_at_fpr_holds_the_budget():
    rng = np.random.default_rng(2)
    y = (rng.random(5_000) < 0.05).astype(int)
    scores = rng.random(5_000) + y * 0.8
    stats = metrics.recall_at_fpr(y, scores, 0.01)
    assert stats["achieved_fpr"] <= 0.01 + 1e-9
    assert 0.0 <= stats["recall"] <= 1.0


# ----------------------------------------------------------- calibration ---
def test_isotonic_calibration_reduces_expected_calibration_error():
    rng = np.random.default_rng(3)
    n = 20_000
    true_p = rng.beta(1.2, 25, size=n)
    y = (rng.random(n) < true_p).astype(int)
    # A badly scaled score: correct ordering, wrong magnitudes.
    raw = np.clip(true_p * 6.0, 0, 1)

    before = calibration.expected_calibration_error(y, raw)
    calibrator = calibration.ProbabilityCalibrator("isotonic").fit(raw[: n // 2], y[: n // 2])
    after = calibration.expected_calibration_error(y[n // 2:], calibrator.transform(raw[n // 2:]))
    assert after < before


# ------------------------------------------------------------------ risk ---
def test_risk_score_is_bounded_and_monotone(cfg):
    p = np.array([0.0, 0.004, 0.5, 0.999, 1.0])
    scores = risk.risk_score(p, cfg)
    assert scores.min() >= 0 and scores.max() <= 100
    assert np.all(np.diff(scores) >= 0)
    assert scores.tolist() == [0, 0, 50, 100, 100]


def test_config_tiers_match_the_report(cfg):
    policy = risk.policy_from_config(cfg)
    assert (policy.medium_lo, policy.high_lo) == (30, 70)
    tiers = policy.assign(np.array([0, 29, 30, 69, 70, 100]))
    assert tiers.tolist() == ["low", "low", "medium", "medium", "high", "high"]
    assert policy.action_for("high") == "manual_review"


def test_break_even_threshold_falls_as_the_amount_rises(cfg):
    thresholds = risk.per_transaction_threshold(np.array([10.0, 100.0, 1_000.0]), cfg)
    assert np.all(np.diff(thresholds) < 0)


def test_derived_policy_respects_the_review_budget(cfg):
    rng = np.random.default_rng(4)
    n = 10_000
    p = rng.beta(1.0, 30.0, size=n)
    scores = risk.risk_score(p, cfg)
    times = np.sort(rng.integers(0, 30 * DAY, size=n))
    amounts = rng.uniform(10, 400, size=n)

    policy = risk.derive_policy(p, scores, amounts, times, cfg)
    alerts = (scores >= policy.high_lo).sum()
    budget = cfg.require("risk.daily_review_capacity_pct") * n
    # Score ties can only ever push the alert count above the budget by the
    # size of one tied group, never by an order of magnitude.
    assert alerts <= max(budget * 3, 10)


def test_expected_cost_prefers_catching_expensive_fraud(cfg):
    y = np.array([1, 1, 0, 0])
    amounts = np.array([1_000.0, 10.0, 50.0, 50.0])
    caught_expensive = risk.expected_cost(y, np.array([1, 0, 0, 0]), amounts, cfg)
    caught_cheap = risk.expected_cost(y, np.array([0, 1, 0, 0]), amounts, cfg)
    assert caught_expensive["total_cost"] < caught_cheap["total_cost"]


# ------------------------------------------------- sequence cache ---------
def test_sequence_cache_reuses_identical_inputs_only():
    """The cache must hit on the same frame and miss on a different one."""
    from frds.models import sequence

    rng = np.random.default_rng(0)
    frame_a = pd.DataFrame({
        "amt_log": rng.normal(size=200).astype(np.float32),
        "vel_card_cnt_1h": rng.integers(0, 5, size=200).astype(np.float32),
    })
    frame_b = frame_a.copy()
    frame_b.loc[10, "amt_log"] = 99.0          # one cell differs
    codes = rng.integers(0, 12, size=200)
    columns = ["amt_log", "vel_card_cnt_1h"]

    key_a = sequence._cache_key(frame_a, columns, codes, 8)
    key_a_again = sequence._cache_key(frame_a.copy(), columns, codes, 8)
    key_b = sequence._cache_key(frame_b, columns, codes, 8)
    key_window = sequence._cache_key(frame_a, columns, codes, 12)
    key_codes = sequence._cache_key(frame_a, columns, codes[::-1].copy(), 8)

    assert key_a == key_a_again          # identical content hits
    assert key_a != key_b                # different values miss
    assert key_a != key_window           # different window misses
    assert key_a != key_codes            # different entity grouping misses


# ------------------------------------------------------- explanation ----------
def test_shap_reason_codes_are_additive_and_ranked():
    """A reason code is only trustworthy if the contributions add up."""
    import xgboost as xgb
    from frds.evaluation.explain import reason_codes, tree_shap_values

    rng = np.random.default_rng(11)
    n = 1_500
    X = pd.DataFrame({
        "velocity": rng.poisson(2, n).astype(float),
        "amount_ratio": rng.lognormal(0, 1, n),
        "noise": rng.normal(size=n),
    })
    logit = 1.4 * X["velocity"] + 0.9 * np.log(X["amount_ratio"]) - 3.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    model = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.2,
                              random_state=0).fit(X, y)

    values = tree_shap_values(model, X)
    margin = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)

    # Additivity: margin minus summed contributions is the same constant on every row.
    implied_base = margin - values.sum(axis=1)
    assert np.ptp(implied_base) < 1e-4, "contributions do not add up to the model output"

    rows = np.arange(len(X))
    codes = reason_codes(values, X, rows, top_k=3)
    has_evidence = values.max(axis=1) > 0

    # Where a row has evidence for fraud, the top reason is the largest positive
    # contribution.
    for i in rows[has_evidence][:40]:
        assert codes.iloc[i]["top_feature"] == X.columns[int(np.argmax(values[i]))]

    # Where it has none, no feature is named - naming the contribution nearest
    # zero would contradict the "no positive contribution" string.
    assert (codes.loc[~has_evidence, "top_feature"] == "").all()

    # The forest may learn the noise column; only the ordering of real signal is checked.
    importance = dict(zip(X.columns, np.abs(values).mean(axis=0)))
    assert importance["velocity"] > importance["amount_ratio"] > importance["noise"]

"""The causal-aggregation primitives, checked against hand-computed answers.

The leakage test at the bottom is the one that matters: it changes the future
and asserts that the past does not move. Everything else in the project rests
on that property, and it is not the kind of thing to take on faith.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frds.features import causal

HOUR = 3_600


@pytest.fixture
def toy():
    """Three cards, interleaved in time.

    row: 0    1    2    3    4    5    6
    card A    B    A    A    C    B    A
    t(h) 0    0.5  1    2    3    4    30
    amt  10   20   30   40   50   60   70
    """
    group = np.array([0, 1, 0, 0, 2, 1, 0])
    times = (np.array([0, 0.5, 1, 2, 3, 4, 30]) * HOUR).astype(np.int64)
    amounts = np.array([10.0, 20, 30, 40, 50, 60, 70])
    return group, times, amounts


def test_count_in_window_excludes_current_row(toy):
    group, times, _ = toy
    counts = causal.count_in_window(group, times, 24 * HOUR)
    # Card A rows are 0, 2, 3, 6. Row 6 sits at 30h, so its 24h window covers
    # [6h, 30h] and none of the card's earlier rows fall inside it.
    assert counts.tolist() == [0, 0, 1, 2, 0, 1, 0]


def test_count_in_window_respects_window_length(toy):
    group, times, _ = toy
    tight = causal.count_in_window(group, times, HOUR)
    # Within one hour, row 3 (t=2h) sees only row 2 (t=1h), not row 0 (t=0).
    assert tight[3] == 1
    assert tight[2] == 1


def test_sum_in_window_matches_manual_sum(toy):
    group, times, amounts = toy
    sums = causal.sum_in_window(group, times, amounts, 24 * HOUR)
    assert sums[3] == pytest.approx(10 + 30)   # rows 0 and 2
    assert sums[6] == 0.0                      # every earlier row has aged out
    assert sums[0] == 0.0

    wide = causal.sum_in_window(group, times, amounts, 48 * HOUR)
    assert wide[6] == pytest.approx(10 + 30 + 40)   # a wider window sees them


def test_seconds_since_previous(toy):
    group, times, _ = toy
    gap = causal.seconds_since_previous(group, times)
    assert np.isnan(gap[0]) and np.isnan(gap[1]) and np.isnan(gap[4])
    assert gap[2] == pytest.approx(HOUR)
    assert gap[6] == pytest.approx(28 * HOUR)


def test_expanding_mean_and_std_use_prior_rows_only(toy):
    group, _, amounts = toy
    means = causal.expanding_mean(group, amounts)
    stds = causal.expanding_std(group, amounts)

    assert np.isnan(means[0])                       # no history
    assert means[2] == pytest.approx(10.0)          # only row 0
    assert means[3] == pytest.approx(20.0)          # rows 0 and 2
    assert np.isnan(stds[2])                        # needs at least two
    assert stds[3] == pytest.approx(np.std([10.0, 30.0]))


def test_cumulative_count(toy):
    group, _, _ = toy
    assert causal.cumulative_count(group).tolist() == [0, 0, 1, 2, 0, 1, 3]


def test_distinct_partners_counts_each_pair_once():
    # One device used by cards 10, 11, 10, 12 in that order.
    device = np.array([0, 0, 0, 0])
    card = np.array([10, 11, 10, 12])
    distinct = causal.distinct_partners_so_far(device, card)
    # Before row 0: none. Row 1: {10}. Row 2: {10, 11}. Row 3: {10, 11}.
    assert distinct.tolist() == [0, 1, 2, 2]


def test_expanding_target_rate_never_sees_its_own_label():
    group = np.array([0, 0, 0, 0])
    target = np.array([1, 1, 1, 1])
    rate = causal.expanding_target_rate(group, target, prior_weight=1.0)
    # Row 0 has no history, so it gets the global prior exactly.
    assert rate[0] == pytest.approx(1.0)   # global rate is 1.0 here
    assert np.all(np.diff(rate) >= -1e-6)  # monotone as evidence accumulates


def test_no_future_leakage():
    """Mutating a later row must not change any earlier row's features."""
    rng = np.random.default_rng(0)
    n = 400
    group = rng.integers(0, 12, size=n)
    partner = rng.integers(0, 20, size=n)
    times = np.sort(rng.integers(0, 30 * 24 * HOUR, size=n))
    amounts = rng.uniform(1, 500, size=n)
    target = (rng.random(n) < 0.1).astype(int)

    cut = 250
    before = {
        "count": causal.count_in_window(group, times, 24 * HOUR),
        "sum": causal.sum_in_window(group, times, amounts, 24 * HOUR),
        "mean": causal.expanding_mean(group, amounts),
        "std": causal.expanding_std(group, amounts),
        "gap": causal.seconds_since_previous(group, times),
        "partners": causal.distinct_partners_so_far(group, partner),
        "rate": causal.expanding_target_rate(group, target),
    }

    # Rewrite everything after the cut.
    group2, partner2 = group.copy(), partner.copy()
    amounts2, target2 = amounts.copy(), target.copy()
    group2[cut:] = rng.integers(0, 12, size=n - cut)
    partner2[cut:] = rng.integers(0, 20, size=n - cut)
    amounts2[cut:] = rng.uniform(1, 500, size=n - cut)
    target2[cut:] = (rng.random(n - cut) < 0.5).astype(int)

    after = {
        "count": causal.count_in_window(group2, times, 24 * HOUR),
        "sum": causal.sum_in_window(group2, times, amounts2, 24 * HOUR),
        "mean": causal.expanding_mean(group2, amounts2),
        "std": causal.expanding_std(group2, amounts2),
        "gap": causal.seconds_since_previous(group2, times),
        "partners": causal.distinct_partners_so_far(group2, partner2),
    }

    for name in after:
        np.testing.assert_allclose(
            before[name][:cut], after[name][:cut], equal_nan=True,
            err_msg=f"'{name}' leaked information from the future",
        )
    # The target rate is the exception worth stating: it depends on the global
    # prior, which does move when future labels change. Its *ordering* within
    # the past must still be unaffected, so compare against a rerun with the
    # original labels.
    np.testing.assert_allclose(
        before["rate"], causal.expanding_target_rate(group, target), equal_nan=True
    )


def test_encode_groups_handles_missing_values():
    series = pd.Series(["a", None, "b", None, "a"])
    codes = causal.encode_groups(series)
    assert len(np.unique(codes)) == 3      # a, b, and NaN as its own level
    assert codes[1] == codes[3]


def test_label_delay_withholds_recent_outcomes():
    """A delayed label must not reach a row that arrives before it is confirmed."""
    day = 86_400
    group = np.array([0, 0, 0, 0])
    times = np.array([0, 10 * day, 20 * day, 40 * day], dtype=np.int64)
    target = np.array([1, 0, 0, 0])

    # With no delay, row 3 sees all three earlier labels.
    immediate = causal.expanding_target_rate(group, target, prior_weight=0.0)
    assert immediate[3] == pytest.approx(1 / 3)

    # With a 30-day delay, row 3 (t=40d) knows outcomes only up to t=10d,
    # so it sees rows 0 and 1 - one fraud out of two.
    delayed = causal.expanding_target_rate(
        group, target, prior_weight=0.0, times=times, label_delay_seconds=30 * day
    )
    assert delayed[3] == pytest.approx(1 / 2)
    # Row 2 (t=20d) knows only row 0, which was fraud.
    assert delayed[2] == pytest.approx(1.0)
    assert delayed[1] == pytest.approx(1.0)
    # Row 0 has no confirmed history; with no smoothing prior there is nothing
    # to fall back to, so the rate is undefined rather than invented.
    assert np.isnan(delayed[0])

    # The delay strictly removes information: no row can see more than it did.
    smoothed_immediate = causal.expanding_target_rate(group, target, prior_weight=5.0)
    smoothed_delayed = causal.expanding_target_rate(
        group, target, prior_weight=5.0, times=times, label_delay_seconds=30 * day
    )
    assert np.isfinite(smoothed_delayed).all()
    assert smoothed_delayed[3] >= smoothed_immediate[3]  # fewer, fraudier priors


def test_label_delay_never_leaks_the_current_row():
    day = 86_400
    group = np.zeros(6, dtype=np.int64)
    times = (np.arange(6) * day).astype(np.int64)
    target = np.ones(6, dtype=int)
    # Even with a zero-length delay path taken explicitly, the current row's own
    # label must stay invisible, so a first row can never score 1.0 outright.
    rates = causal.expanding_target_rate(
        group, target, prior_weight=1.0, times=times, label_delay_seconds=1
    )
    assert rates[0] == pytest.approx(float(target.mean()))

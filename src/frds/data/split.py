"""Chronological splitting (report objective 1, gap G1).

Nothing in this project uses a random split. The timeline is cut once into
train / valid / test, and the out-of-fold predictions the stacker consumes come
from expanding-window folds inside the train span - never from shuffled folds,
which would let a card's future behaviour inform its own past score.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger

log = get_logger("frds.data.split")


@dataclass(frozen=True)
class SplitIndex:
    """Positional indices into a time-sorted frame."""

    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    cut_train_valid: float
    cut_valid_test: float

    def summary(self, df: pd.DataFrame, time_col: str, target_col: str) -> dict:
        out = {}
        for name in ("train", "valid", "test"):
            idx = getattr(self, name)
            part = df.iloc[idx]
            out[name] = {
                "rows": int(len(idx)),
                "fraud": int(part[target_col].sum()),
                "prevalence": float(part[target_col].mean()),
                "t_start_day": float(part[time_col].min() / 86_400),
                "t_end_day": float(part[time_col].max() / 86_400),
            }
        out["cut_train_valid_day"] = self.cut_train_valid / 86_400
        out["cut_valid_test_day"] = self.cut_valid_test / 86_400
        return out


def chronological_split(df: pd.DataFrame, cfg: Config) -> SplitIndex:
    """Cut the timeline at the configured quantiles of TransactionDT.

    The cut is made on the *time value*, not on row position, so every
    transaction sharing a timestamp lands on the same side of the boundary.
    """
    time_col = cfg.require("data.time_col")
    train_frac = float(cfg.require("split.train_frac"))
    valid_frac = float(cfg.require("split.valid_frac"))
    test_frac = float(cfg.require("split.test_frac"))

    total = train_frac + valid_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    t = df[time_col].to_numpy()
    if not np.all(np.diff(t) >= 0):
        raise ValueError("frame must be sorted by TransactionDT before splitting")

    cut_tv = float(np.quantile(t, train_frac))
    cut_vt = float(np.quantile(t, train_frac + valid_frac))

    train = np.flatnonzero(t <= cut_tv)
    valid = np.flatnonzero((t > cut_tv) & (t <= cut_vt))
    test = np.flatnonzero(t > cut_vt)

    for name, idx in (("train", train), ("valid", valid), ("test", test)):
        if idx.size == 0:
            raise ValueError(f"{name} split is empty; check split fractions")

    split = SplitIndex(train, valid, test, cut_tv, cut_vt)
    target = cfg.require("data.target_col")
    for name in ("train", "valid", "test"):
        part = df.iloc[getattr(split, name)]
        log.info(
            "%-5s rows=%7d fraud=%6d prevalence=%.3f%% days %.1f-%.1f",
            name, len(part), int(part[target].sum()), 100 * part[target].mean(),
            part[time_col].min() / 86_400, part[time_col].max() / 86_400,
        )
    return split


def expanding_window_folds(
    time_values: np.ndarray, n_folds: int, min_train_frac: float = 0.35
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (fit_idx, score_idx) pairs over an expanding time window.

    Fold k fits on everything before boundary k and scores the slice between
    boundary k and boundary k+1. Concatenating the score slices gives one
    out-of-fold prediction per row in the tail of the train span, which is
    exactly what the stacking meta-learner needs.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    n = len(time_values)
    start = int(n * min_train_frac)
    boundaries = np.linspace(start, n, n_folds + 1).astype(int)

    for k in range(n_folds):
        lo, hi = boundaries[k], boundaries[k + 1]
        if hi <= lo:
            continue
        # Push the boundary forward past any tie in the timestamp so a single
        # instant never straddles fit and score.
        while lo < n and lo > 0 and time_values[lo] == time_values[lo - 1]:
            lo += 1
        if hi <= lo:
            continue
        yield np.arange(0, lo), np.arange(lo, hi)


def temporal_windows(time_values: np.ndarray, n_windows: int) -> list[np.ndarray]:
    """Split an index range into ``n_windows`` equal-duration slices.

    Used by the PSI drift monitor, which compares successive deployment
    periods rather than successive equal-count batches.
    """
    t_min, t_max = float(time_values.min()), float(time_values.max())
    edges = np.linspace(t_min, t_max, n_windows + 1)
    windows = []
    for k in range(n_windows):
        lo, hi = edges[k], edges[k + 1]
        mask = (time_values >= lo) & (time_values < hi) if k < n_windows - 1 \
            else (time_values >= lo) & (time_values <= hi)
        windows.append(np.flatnonzero(mask))
    return windows


def save_split(split: SplitIndex, cfg: Config, df: pd.DataFrame) -> None:
    from ..utils import write_json

    payload = split.summary(
        df, cfg.require("data.time_col"), cfg.require("data.target_col")
    )
    payload["strategy"] = cfg.require("split.strategy")
    write_json(payload, cfg.path("reports") / "metrics" / "split_summary.json")
    np.savez_compressed(
        cfg.path("interim") / "split_index.npz",
        train=split.train, valid=split.valid, test=split.test,
        cut_train_valid=split.cut_train_valid, cut_valid_test=split.cut_valid_test,
    )


def load_split(cfg: Config) -> SplitIndex:
    path = cfg.path("interim") / "split_index.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m frds features' first.")
    z = np.load(path)
    return SplitIndex(
        train=z["train"], valid=z["valid"], test=z["test"],
        cut_train_valid=float(z["cut_train_valid"]),
        cut_valid_test=float(z["cut_valid_test"]),
    )

"""Sequential tier: LSTM and temporal convolutional network.

Report Table 4.2. The premise is that fraud is a *break in a sequence* rather
than a property of an isolated row, so this tier sees a card's last K
transactions in order and is asked whether the current one fits the pattern.

Sequence construction is causal by design. Rows are grouped by entity while
preserving time order, so the K rows preceding a row inside its own group are
contiguous in the sorted arrangement and can be gathered without a Python loop.
Positions earlier than the entity's first transaction are zero-padded and
masked, so a card's first transaction is never given a fabricated history.

Only a compact behavioural feature set feeds this tier - amount, recency,
velocity, hour of day. Feeding it all 255 columns would just rebuild the
boosting tier inside an RNN and destroy the point of the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseScorer, FitContext, log
from .torch_utils import StandardScalerNP, get_device, predict_in_batches, train_loop

#: Column name prefixes that describe behaviour over time.
SEQUENCE_PREFIXES = ("amt_", "vel_", "recency_", "roll_", "txn_hour", "txn_dow")


def select_sequence_columns(X: pd.DataFrame, max_columns: int = 24) -> list[str]:
    chosen = [c for c in X.columns if c.startswith(SEQUENCE_PREFIXES)]
    return chosen[:max_columns]


def build_history_index(
    group_codes: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (index_matrix, mask) of shape (n, window).

    ``index_matrix[i, j]`` is the row position of the j-th step of row i's
    history, oldest first, with the current row as the final step. ``mask`` is
    False where the entity had no transaction that far back.
    """
    n = len(group_codes)
    order = np.argsort(group_codes, kind="stable")
    g = group_codes[order]

    starts = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]]))
    group_start = np.repeat(starts, np.diff(np.concatenate([starts, [n]])))
    pos_in_group = np.arange(n) - group_start

    sorted_positions = np.arange(n)
    index_sorted = np.empty((n, window), dtype=np.int64)
    mask_sorted = np.empty((n, window), dtype=bool)

    for j in range(window):
        lag = window - 1 - j                      # j = window-1 is the current row
        available = pos_in_group >= lag
        source = np.where(available, sorted_positions - lag, group_start)
        index_sorted[:, j] = order[source]
        mask_sorted[:, j] = available

    index_matrix = np.empty_like(index_sorted)
    mask = np.empty_like(mask_sorted)
    index_matrix[order] = index_sorted
    mask[order] = mask_sorted
    return index_matrix, mask



#: Built once per (frame, entity, window) and reused across folds. Keyed by
#: content rather than by object identity, because the experiment scripts run
#: several different feature matrices inside one process and a stale hit would
#: silently score one cell with another cell's features.
_SEQUENCE_CACHE: dict[tuple, dict] = {}


def _cache_key(
    full: pd.DataFrame, columns: list[str], codes: np.ndarray, window: int
) -> tuple:
    """Cheap content fingerprint of the inputs the sequence cache depends on."""
    import hashlib

    step = max(1, len(full) // 512)
    sample = full[columns].to_numpy(np.float32)[::step]
    digest = hashlib.blake2b(np.ascontiguousarray(sample).tobytes(), digest_size=16)
    digest.update(np.ascontiguousarray(codes[::step]).tobytes())
    return (tuple(columns), int(window), len(full), digest.hexdigest())


def _build_network(kind: str, n_features: int, window: int, hidden: int, dropout: float):
    import torch
    from torch import nn

    class LSTMHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rnn = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden + n_features, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, seq, mask):
            out, _ = self.rnn(seq)
            # Padding sits at the *front* of the window, so the final step is
            # always the current transaction and is always valid.
            summary = out[:, -1, :]
            return self.head(torch.cat([summary, seq[:, -1, :]], dim=1))

    class TCNHead(nn.Module):
        """Dilated causal convolutions - same receptive field, no recurrence."""

        def __init__(self) -> None:
            super().__init__()
            layers = []
            channels = n_features
            for dilation in (1, 2, 4):
                layers += [
                    nn.Conv1d(channels, hidden, kernel_size=2, dilation=dilation,
                              padding=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                channels = hidden
            self.conv = nn.Sequential(*layers)
            self.head = nn.Sequential(
                nn.Linear(hidden + n_features, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )

        def forward(self, seq, mask):
            # (batch, steps, features) -> (batch, features, steps)
            z = self.conv(seq.transpose(1, 2))[:, :, :seq.size(1)]
            return self.head(torch.cat([z[:, :, -1], seq[:, -1, :]], dim=1))

    return LSTMHead() if kind == "lstm" else TCNHead()


class SequenceScorer(BaseScorer):
    """Shared implementation; ``kind`` picks the recurrent or convolutional head."""

    name = "sequence"
    kind = "lstm"
    handles_missing = True

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.window = int(self.params.get("window", 12))
        self.hidden = int(self.params.get("hidden", 64))
        self.dropout = float(self.params.get("dropout", 0.2))
        self.entity = str(self.params.get("entity", "uid"))
        self.epochs = int(self.params.get("epochs", 12))
        self.batch_size = int(self.params.get("batch_size", 1024))
        self.lr = float(self.params.get("lr", 1e-3))
        self.sequence_columns: list[str] = []
        self.scaler = StandardScalerNP()
        self._cache: dict = {}

    # -- context -------------------------------------------------------
    def _prepare(self, ctx: FitContext) -> None:
        if ctx.full_X is None or ctx.entity_keys is None:
            raise ValueError(
                f"{self.name} needs the full frame and entity keys; the pipeline "
                "must pass them through FitContext."
            )
        full = ctx.full_X
        self.sequence_columns = select_sequence_columns(full)
        codes = ctx.entity_keys[self.entity]

        # The history index and the scaled value matrix depend only on the
        # frame, the entity and the window - not on which fold is being fitted.
        # A run fits this tier seven times (five out-of-fold folds, the final
        # fit and the refit), and rebuilding an identical 590,540 x window
        # index each time costs minutes per fit for no benefit.
        key = _cache_key(full, self.sequence_columns, codes, self.window)
        cached = _SEQUENCE_CACHE.get(key)
        if cached is not None:
            self._cache = cached["cache"]
            self.scaler = cached["scaler"]
            return

        values = full[self.sequence_columns].to_numpy(np.float32)
        scaled = self.scaler.fit_transform(values)
        index_matrix, mask = build_history_index(codes, self.window)
        self._cache = {"values": scaled, "index": index_matrix, "mask": mask}

        # One entry is enough: a run only ever works on one frame at a time, and
        # holding several 590k x 24 matrices at once is what would actually hurt.
        _SEQUENCE_CACHE.clear()
        _SEQUENCE_CACHE[key] = {"cache": self._cache, "scaler": self.scaler}
        log.info(
            "%s: %d sequence features, window %d over entity '%s'",
            self.name, len(self.sequence_columns), self.window, self.entity,
        )

    def _batch(self, positions: np.ndarray, device):
        import torch

        cache = self._cache
        idx = cache["index"][positions]
        seq = cache["values"][idx]
        mask = cache["mask"][positions]
        seq = seq * mask[:, :, None]  # zero out padded steps
        return (
            torch.tensor(seq, dtype=torch.float32, device=device),
            torch.tensor(mask, dtype=torch.float32, device=device),
        )

    # -- fit / predict --------------------------------------------------
    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        import torch

        self._remember_columns(X_train)
        self._prepare(ctx)
        torch.manual_seed(ctx.seed)

        if len(ctx.full_X) <= int(X_train.index.max()):
            raise ValueError(
                f"{self.name} locates rows through X.index; a resampled fold "
                "destroys that mapping. Use imbalance.method: class_weight."
            )

        device = get_device()
        self.model = _build_network(
            self.kind, len(self.sequence_columns), self.window, self.hidden, self.dropout
        )

        train_pos = X_train.index.to_numpy()
        if X_valid is not None and len(X_valid):
            valid_pos = X_valid.index.to_numpy()
            y_all = np.zeros(len(ctx.full_X), dtype=np.float32)
            y_all[train_pos] = y_train
            y_all[valid_pos] = y_valid
        else:
            # No validation slice: hold out the final 10% of the fit window,
            # which keeps the holdout chronologically after the training rows.
            cut = int(len(train_pos) * 0.9)
            valid_pos = train_pos[cut:]
            train_pos = train_pos[:cut]
            y_all = np.zeros(len(ctx.full_X), dtype=np.float32)
            y_all[X_train.index.to_numpy()] = y_train

        def forward_batch(positions, training: bool):
            seq, mask = self._batch(positions, device)
            return self.model(seq, mask)

        stats = train_loop(
            self.model, forward_batch, train_pos, valid_pos, y_all,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            pos_weight=ctx.class_weight, device=device, label=self.name,
        )
        self.params["training"] = stats["history"][-1] if stats["history"] else {}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        device = get_device()
        # A model reloaded from disk lands on the CPU; move it before scoring.
        self.model.to(device)
        positions = X.index.to_numpy()

        def forward_batch(batch, training: bool):
            seq, mask = self._batch(batch, device)
            return self.model(seq, mask)

        return predict_in_batches(
            self.model, forward_batch, positions, self.batch_size, device
        )

    # -- persistence ----------------------------------------------------
    def save(self, directory):
        from pathlib import Path

        import joblib

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.joblib"
        joblib.dump(
            {
                "params": self.params,
                "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                "features": self.feature_names_,
                "sequence_columns": self.sequence_columns,
                "scaler": self.scaler,
                "kind": self.kind,
                "entity": self.entity,
                "window": self.window,
                "hidden": self.hidden,
                "dropout": self.dropout,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path):
        """Rebuild the network from its state dict.

        The sequence cache is *not* stored - it is a view over the whole
        feature frame and would double the artifact size. Call
        ``restore_context`` with the frame before scoring.
        """
        import joblib

        payload = joblib.load(path)
        obj = cls(payload["params"])
        obj.feature_names_ = payload["features"]
        obj.sequence_columns = payload["sequence_columns"]
        obj.scaler = payload["scaler"]
        obj.entity = payload["entity"]
        obj.window = int(payload["window"])
        obj.hidden = int(payload["hidden"])
        obj.dropout = float(payload["dropout"])
        obj.model = _build_network(
            payload["kind"], len(obj.sequence_columns), obj.window, obj.hidden, obj.dropout
        )
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        return obj

    def restore_context(self, full_X: pd.DataFrame, entity_keys: dict) -> "SequenceScorer":
        """Rebuild the per-row history index a reloaded model needs."""
        values = self.scaler.transform(full_X[self.sequence_columns].to_numpy(np.float32))
        index_matrix, mask = build_history_index(entity_keys[self.entity], self.window)
        self._cache = {"values": values, "index": index_matrix, "mask": mask}
        return self


class LSTMScorer(SequenceScorer):
    name = "lstm"
    kind = "lstm"


class TCNScorer(SequenceScorer):
    name = "tcn"
    kind = "tcn"

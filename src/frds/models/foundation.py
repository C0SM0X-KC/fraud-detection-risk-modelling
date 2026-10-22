"""Foundation tier: TabPFN v2 as a zero-training tabular baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseScorer, FitContext, log

INSTALL_HINT = (
    "TabPFN is not installed. Run 'pip install tabpfn' (it downloads model "
    "weights on first use), or remove 'tabpfn' from models.enabled."
)


class TabPFNScorer(BaseScorer):
    name = "tabpfn"
    handles_missing = True

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.context_size = int(self.params.get("context_size", 8_000))
        self.max_features = int(self.params.get("max_features", 100))
        self.negatives_per_positive = float(self.params.get("negatives_per_positive", 8.0))
        self.batch_size = int(self.params.get("batch_size", 2_000))
        self.selected_features: list[str] = []
        self.prior_correction_: float = 1.0

    # -- context construction ------------------------------------------
    def _select_features(self, X: pd.DataFrame, y: np.ndarray) -> list[str]:
        from sklearn.feature_selection import mutual_info_classif

        if X.shape[1] <= self.max_features:
            return list(X.columns)

        # Mutual information on a subsample: computing it on 400k rows costs
        # more than the model fit it is preparing for.
        rng = np.random.default_rng(0)
        sample = rng.choice(len(X), size=min(20_000, len(X)), replace=False)
        filled = X.iloc[sample].fillna(X.iloc[sample].median(numeric_only=True)).fillna(0.0)
        scores = mutual_info_classif(filled, y[sample], random_state=0)
        order = np.argsort(scores)[::-1][: self.max_features]
        return [X.columns[i] for i in sorted(order)]

    def _subsample_context(self, X: pd.DataFrame, y: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)

        n_pos = min(len(positives), max(1, int(self.context_size / (1 + self.negatives_per_positive))))
        n_neg = min(len(negatives), int(n_pos * self.negatives_per_positive))

        pos_idx = rng.choice(positives, size=n_pos, replace=False)
        neg_idx = rng.choice(negatives, size=n_neg, replace=False)
        idx = np.sort(np.concatenate([pos_idx, neg_idx]))

        original_odds = len(positives) / max(len(negatives), 1)
        context_odds = n_pos / max(n_neg, 1)
        self.prior_correction_ = original_odds / max(context_odds, 1e-12)

        log.info(
            "tabpfn context: %d rows (%d positive), prior odds %.5f -> %.5f "
            "(correction factor %.4f)",
            len(idx), n_pos, original_odds, context_odds, self.prior_correction_,
        )
        return X.iloc[idx], y[idx]

    # -- fit / predict ---------------------------------------------------
    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(INSTALL_HINT) from exc

        self._remember_columns(X_train)
        y_train = np.asarray(y_train)
        self.selected_features = self._select_features(X_train, y_train)

        X_context, y_context = self._subsample_context(
            X_train[self.selected_features], y_train, ctx.seed
        )

        device = "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        self.model = TabPFNClassifier(device=device, **self.params.get("model_kwargs", {}))
        log.info("fitting %s in-context on %s (device=%s)", self.name, X_context.shape, device)
        # "fit" here stores the context; no gradient step is taken.
        self.model.fit(X_context.to_numpy(np.float32), y_context)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = self._align(X)[self.selected_features].to_numpy(np.float32)
        out = np.empty(len(X), dtype=np.float64)
        for start in range(0, len(X), self.batch_size):
            chunk = X[start: start + self.batch_size]
            out[start: start + len(chunk)] = self.model.predict_proba(chunk)[:, 1]
        return self._correct_prior(out)

    def _correct_prior(self, p: np.ndarray) -> np.ndarray:
        """Undo the class-balance shift introduced by context subsampling."""
        p = np.clip(p, 1e-9, 1 - 1e-9)
        odds = p / (1 - p) * self.prior_correction_
        return odds / (1 + odds)

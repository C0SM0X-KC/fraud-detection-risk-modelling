"""Shared PyTorch plumbing for the sequence and graph tiers.

Kept deliberately small: device selection, a standardising scaler that lives
with the model, and one training loop with early stopping on validation AUC-PR.
Both deep tiers use the same loop, so a difference in their results is a
difference in the model rather than in how hard each was trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("frds.models.torch")


def get_device(prefer_gpu: bool = True):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class StandardScalerNP:
    """Median-impute then standardise. Fitted statistics travel with the model.

    sklearn's StandardScaler would do, but it does not impute, and a NaN
    reaching a tensor turns the whole loss into NaN with no useful error.
    """

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    fill_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScalerNP":
        X = np.asarray(X, dtype=np.float32)
        self.fill_ = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0).astype(np.float32)
        filled = np.where(np.isfinite(X), X, self.fill_)
        self.mean_ = filled.mean(axis=0).astype(np.float32)
        std = filled.std(axis=0).astype(np.float32)
        self.scale_ = np.where(std > 1e-6, std, 1.0).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        filled = np.where(np.isfinite(X), X, self.fill_)
        return np.clip((filled - self.mean_) / self.scale_, -10.0, 10.0).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def auc_pr(y_true: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def train_loop(
    model,
    forward_batch,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 12,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    pos_weight: float = 1.0,
    patience: int = 3,
    device=None,
    shuffle: bool = True,
    label: str = "model",
) -> dict:
    """Train with BCE-on-logits, early-stopping on validation AUC-PR.

    ``forward_batch(indices) -> logits`` is supplied by the caller, which is
    what lets the sequence tier and the graph tier share this loop despite
    assembling their inputs completely differently.

    ``shuffle=False`` matters for the graph tier: its entity memories are
    updated in time order, so batches must arrive in time order too.
    """
    import torch

    device = device or get_device()
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
    )

    best_score, best_state, bad_epochs = -np.inf, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(train_indices) if shuffle else train_indices
        total_loss, n_batches = 0.0, 0

        for start in range(0, len(order), batch_size):
            batch = order[start: start + batch_size]
            targets = torch.tensor(y[batch], dtype=torch.float32, device=device)
            optimiser.zero_grad(set_to_none=True)
            logits = forward_batch(batch, training=True)
            loss = criterion(logits.squeeze(-1), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total_loss += float(loss.detach())
            n_batches += 1

        val_scores = predict_in_batches(model, forward_batch, valid_indices, batch_size, device)
        score = auc_pr(y[valid_indices], val_scores)
        history.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1),
                        "valid_auc_pr": score})
        log.info("%s epoch %d/%d loss %.4f valid AUC-PR %.4f",
                 label, epoch, epochs, history[-1]["loss"], score)

        if np.isfinite(score) and score > best_score + 1e-5:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("%s early stopped at epoch %d", label, epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_valid_auc_pr": best_score, "history": history}


def predict_in_batches(model, forward_batch, indices: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch

    model.eval()
    out = np.empty(len(indices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start: start + batch_size]
            logits = forward_batch(batch, training=False)
            out[start: start + len(batch)] = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    return out

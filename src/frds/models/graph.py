"""Relational tier: heterogeneous graph neural network with attention."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseScorer, FitContext, log
from .torch_utils import StandardScalerNP, get_device, train_loop

#: Entity types that become node types in the heterogeneous graph.
RELATIONS = ("card", "merchant", "device", "address", "email")

#: Transaction-node input features. Behavioural and relational columns only -
#: the 339 raw V columns belong to the tabular tier, not to this one.
NODE_PREFIXES = ("amt_", "vel_", "recency_", "roll_", "graph_", "mismatch_",
                 "txn_hour", "txn_dow", "freq_")


def select_node_columns(X: pd.DataFrame, max_columns: int = 48) -> list[str]:
    chosen = [c for c in X.columns if c.startswith(NODE_PREFIXES)]
    return chosen[:max_columns]


def _build_network(n_features: int, hidden: int, dropout: float, n_relations: int):
    import torch
    from torch import nn

    class HeteroAttentionGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(),
            )
            # One projection and attention vector per relation (what makes it heterogeneous).
            self.relation_proj = nn.ModuleList(
                [nn.Linear(hidden, hidden) for _ in range(n_relations)]
            )
            self.attention = nn.ModuleList(
                [nn.Linear(2 * hidden, 1) for _ in range(n_relations)]
            )
            self.leaky = nn.LeakyReLU(0.2)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def encode(self, x):
            return self.encoder(x)

        def attend(self, h, memories):
            """Attention over the five relations for each transaction."""
            projected, logits = [], []
            for r, memory in enumerate(memories):
                m = self.relation_proj[r](memory)
                projected.append(m)
                logits.append(self.leaky(self.attention[r](torch.cat([h, m], dim=1))))
            weights = torch.softmax(torch.cat(logits, dim=1), dim=1)   # (batch, R)
            stacked = torch.stack(projected, dim=1)                    # (batch, R, hidden)
            return (weights.unsqueeze(-1) * stacked).sum(dim=1), weights

        def forward(self, x, memories):
            h = self.encode(x)
            context, weights = self.attend(h, memories)
            return self.head(torch.cat([h, context], dim=1)), h, weights

    return HeteroAttentionGNN()


class HeteroGNNScorer(BaseScorer):
    """Temporal heterogeneous GNN over the card-merchant-device graph."""

    name = "hetero_gnn"
    handles_missing = True

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        self.hidden = int(self.params.get("hidden", 64))
        self.dropout = float(self.params.get("dropout", 0.2))
        self.epochs = int(self.params.get("epochs", 8))
        self.batch_size = int(self.params.get("batch_size", 2048))
        self.lr = float(self.params.get("lr", 2e-3))
        self.relations = tuple(self.params.get("relations", RELATIONS))
        # How much of an entity's past state survives each new transaction.
        self.memory_decay = float(self.params.get("memory_decay", 0.8))
        self.node_columns: list[str] = []
        self.scaler = StandardScalerNP()
        self._codes: dict[str, np.ndarray] = {}
        self._values: np.ndarray | None = None
        self._memory: list = []
        self._n_rows = 0

    # -- context -------------------------------------------------------
    def _prepare(self, ctx: FitContext) -> None:
        if ctx.full_X is None or ctx.entity_keys is None:
            raise ValueError(
                f"{self.name} needs the full frame and entity keys; the pipeline "
                "must pass them through FitContext."
            )
        missing = [r for r in self.relations if r not in ctx.entity_keys]
        if missing:
            raise ValueError(f"{self.name}: entity keys missing for {missing}")

        self.node_columns = select_node_columns(ctx.full_X)
        self._values = self.scaler.fit_transform(
            ctx.full_X[self.node_columns].to_numpy(np.float32)
        )
        self._codes = {r: ctx.entity_keys[r].astype(np.int64) for r in self.relations}
        self._n_rows = len(ctx.full_X)
        log.info(
            "%s: %d node features, relations %s, nodes %s",
            self.name, len(self.node_columns), list(self.relations),
            {r: int(c.max()) + 1 for r, c in self._codes.items()},
        )

    def _reset_memory(self, device) -> None:
        import torch

        self._memory = [
            torch.zeros(
                int(self._codes[r].max()) + 1, self.hidden, device=device
            )
            for r in self.relations
        ]

    def _forward(self, positions: np.ndarray, device, write_memory: bool = True):
        import torch

        x = torch.tensor(self._values[positions], dtype=torch.float32, device=device)
        index = [
            torch.tensor(self._codes[r][positions], dtype=torch.long, device=device)
            for r in self.relations
        ]
        memories = [self._memory[i][index[i]] for i in range(len(self.relations))]

        logits, h, _ = self.model(x, memories)

        if write_memory:
            # Written after scoring, so a transaction never informs itself.
            message = h.detach()
            for i in range(len(self.relations)):
                self._scatter_update(i, index[i], message)
        return logits

    def _scatter_update(self, relation: int, index, message) -> None:
        """Mean-pool the batch's messages per node, then blend into memory."""
        import torch

        memory = self._memory[relation]
        unique, inverse = torch.unique(index, return_inverse=True)
        pooled = torch.zeros(len(unique), message.shape[1], device=message.device)
        counts = torch.zeros(len(unique), 1, device=message.device)
        pooled.index_add_(0, inverse, message)
        counts.index_add_(0, inverse, torch.ones_like(counts[inverse]))
        pooled = pooled / counts.clamp(min=1.0)

        with torch.no_grad():
            decay = self.memory_decay
            memory[unique] = decay * memory[unique] + (1.0 - decay) * pooled

    # -- fit / predict --------------------------------------------------
    def fit(self, X_train, y_train, ctx: FitContext, X_valid=None, y_valid=None):
        import torch

        self._remember_columns(X_train)
        self._prepare(ctx)
        torch.manual_seed(ctx.seed)

        device = get_device()
        self.model = _build_network(
            len(self.node_columns), self.hidden, self.dropout, len(self.relations)
        )
        self.model.to(device)

        train_pos = np.sort(X_train.index.to_numpy())
        if X_valid is not None and len(X_valid):
            valid_pos = np.sort(X_valid.index.to_numpy())
        else:
            cut = int(len(train_pos) * 0.9)
            valid_pos, train_pos = train_pos[cut:], train_pos[:cut]

        y_all = np.zeros(self._n_rows, dtype=np.float32)
        y_all[X_train.index.to_numpy()] = y_train
        if X_valid is not None and len(X_valid):
            y_all[X_valid.index.to_numpy()] = y_valid

        epoch_state = {"n": 0}

        def forward_batch(positions, training: bool):
            # A new pass over the timeline starts from an empty graph memory.
            if positions[0] == train_pos[0]:
                self._reset_memory(device)
                epoch_state["n"] += 1
            return self._forward(np.asarray(positions), device, write_memory=True)

        stats = train_loop(
            self.model, forward_batch, train_pos, valid_pos, y_all,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            pos_weight=ctx.class_weight, device=device, shuffle=False,
            label=self.name,
        )
        self.params["training"] = stats["history"][-1] if stats["history"] else {}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Replay the whole timeline so the memories are warm, then read off
        the scores for the requested rows.
        """
        import torch

        device = get_device()
        self.model.to(device)
        self.model.eval()
        self._reset_memory(device)

        wanted = np.sort(X.index.to_numpy())
        wanted_set = np.zeros(self._n_rows, dtype=bool)
        wanted_set[wanted] = True
        scores = np.zeros(self._n_rows, dtype=np.float32)

        with torch.no_grad():
            for start in range(0, self._n_rows, self.batch_size):
                batch = np.arange(start, min(start + self.batch_size, self._n_rows))
                logits = self._forward(batch, device, write_memory=True)
                scores[batch] = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        return scores[X.index.to_numpy()]

    def predict_online(self, X: pd.DataFrame) -> np.ndarray:
        """Score rows against the memory as it currently stands."""
        import torch

        device = get_device()
        self.model.eval()
        positions = X.index.to_numpy()
        with torch.no_grad():
            logits = self._forward(positions, device, write_memory=True)
            return torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

    def warm_memory(self, up_to_position: int) -> None:
        """Advance the entity memories through every row before a cut-off."""
        import torch

        device = get_device()
        self.model.to(device)
        self.model.eval()
        self._reset_memory(device)
        with torch.no_grad():
            for start in range(0, int(up_to_position), self.batch_size):
                batch = np.arange(start, min(start + self.batch_size, int(up_to_position)))
                if batch.size:
                    self._forward(batch, device, write_memory=True)

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
                "node_columns": self.node_columns,
                "scaler": self.scaler,
                "relations": self.relations,
                "hidden": self.hidden,
                "dropout": self.dropout,
                "memory_decay": self.memory_decay,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path):
        import joblib

        payload = joblib.load(path)
        obj = cls(payload["params"])
        obj.feature_names_ = payload["features"]
        obj.node_columns = payload["node_columns"]
        obj.scaler = payload["scaler"]
        obj.relations = tuple(payload["relations"])
        obj.hidden = int(payload["hidden"])
        obj.dropout = float(payload["dropout"])
        obj.memory_decay = float(payload["memory_decay"])
        obj.model = _build_network(
            len(obj.node_columns), obj.hidden, obj.dropout, len(obj.relations)
        )
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        return obj

    def restore_context(self, full_X: pd.DataFrame, entity_keys: dict) -> "HeteroGNNScorer":
        """Rebuild the node features and the entity index a reloaded model needs."""
        self._values = self.scaler.transform(
            full_X[self.node_columns].to_numpy(np.float32)
        )
        self._codes = {r: entity_keys[r].astype(np.int64) for r in self.relations}
        self._n_rows = len(full_X)
        self._reset_memory(get_device())
        return self

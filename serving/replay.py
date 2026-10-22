"""Replay the held-out span as an arriving transaction stream."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ReplayStream:
    """A cursor over positions, handed out in chronological batches."""

    positions: np.ndarray
    batch_size: int = 256
    cursor: int = 0

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions)
        if self.positions.size and np.any(np.diff(self.positions) < 0):
            raise ValueError(
                "replay positions must be in chronological order; the graph "
                "tier's memories advance with the stream and cannot be rewound"
            )

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.positions)

    @property
    def progress(self) -> float:
        if not len(self.positions):
            return 1.0
        return min(1.0, self.cursor / len(self.positions))

    def next_batch(self) -> np.ndarray:
        """Return the next slice of positions, or an empty array at the end."""
        if self.exhausted:
            return np.empty(0, dtype=self.positions.dtype)
        end = min(self.cursor + self.batch_size, len(self.positions))
        batch = self.positions[self.cursor: end]
        self.cursor = end
        return batch

    def reset(self) -> None:
        self.cursor = 0


@dataclass
class QueueState:
    """Everything the dashboard accumulates as the stream is consumed."""

    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)
    batch_latency_ms: list[float] = field(default_factory=list)
    processed: int = 0

    def extend(self, batch: pd.DataFrame, elapsed_ms: float) -> None:
        self.decisions = (
            batch if self.decisions.empty
            else pd.concat([self.decisions, batch], ignore_index=True)
        )
        self.processed += len(batch)
        if len(batch):
            self.batch_latency_ms.append(elapsed_ms / len(batch))

    # -- derived views --------------------------------------------------
    def alerts(self) -> pd.DataFrame:
        """Everything the policy did not auto-approve, riskiest first."""
        if self.decisions.empty:
            return self.decisions
        flagged = self.decisions[self.decisions["tier"] != "low"]
        return flagged.sort_values("risk_score", ascending=False)

    def summary(self) -> dict:
        if self.decisions.empty:
            return {
                "processed": 0, "alerts": 0, "fraud_seen": 0, "fraud_caught": 0,
                "recall": float("nan"), "precision": float("nan"),
                "alert_rate": float("nan"), "throughput_tps": float("nan"),
            }

        flagged = self.decisions["tier"] != "low"
        is_fraud = self.decisions["is_fraud"] == 1
        caught = int((flagged & is_fraud).sum())
        n_alerts = int(flagged.sum())
        n_fraud = int(is_fraud.sum())
        mean_ms = float(np.mean(self.batch_latency_ms)) if self.batch_latency_ms else float("nan")

        return {
            "processed": self.processed,
            "alerts": n_alerts,
            "fraud_seen": n_fraud,
            "fraud_caught": caught,
            "recall": caught / n_fraud if n_fraud else float("nan"),
            "precision": caught / n_alerts if n_alerts else float("nan"),
            "alert_rate": n_alerts / self.processed if self.processed else float("nan"),
            "throughput_tps": 1000.0 / mean_ms if mean_ms and np.isfinite(mean_ms) else float("nan"),
        }

    def tier_counts(self) -> pd.DataFrame:
        if self.decisions.empty:
            return pd.DataFrame(columns=["tier", "transactions", "fraud", "precision"])
        rows = []
        for tier in ("low", "medium", "high"):
            part = self.decisions[self.decisions["tier"] == tier]
            fraud = int((part["is_fraud"] == 1).sum())
            rows.append({
                "tier": tier,
                "transactions": len(part),
                "fraud": fraud,
                "precision": fraud / len(part) if len(part) else float("nan"),
            })
        return pd.DataFrame(rows)

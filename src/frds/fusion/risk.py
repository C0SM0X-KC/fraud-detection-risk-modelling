"""Decision layer (L4), part 3: risk score, cost-sensitive tiers, actions.

Report section 4.6 and Table 4.3. Two things happen here.

1. The calibrated probability becomes a bounded integer score,
   ``S = round(100 * p)``, which is what an analyst sees.

2. Thresholds on that score are chosen by cost, not by convention. A
   transaction is worth flagging when the expected loss from allowing it
   exceeds the expected loss from declining it:

       p * C_FN  >  (1 - p) * C_FP

   with ``C_FN`` proportional to the transaction amount and ``C_FP`` a fixed
   review-and-goodwill cost. The resulting boundary is then clipped by the
   analyst review capacity, because a tier structure that produces more
   high-risk alerts than can be reviewed is not operationally usable.

A note that the report's fixed bands make necessary: at 3.5% prevalence a
*calibrated* probability rarely exceeds 0.7, so the literal 70-100 band in
Table 4.3 will hold very few transactions. Both are reported - the fixed
bands as specified, and the cost-and-capacity-derived bands that a real
operation would run - so the gap between the two is visible rather than
quietly papered over.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger

log = get_logger("frds.fusion.risk")

SECONDS_PER_DAY = 86_400


def risk_score(p_calibrated: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Map a calibrated probability to the bounded integer risk score."""
    lo = int(cfg.get("risk.score_min", 0)) if cfg else 0
    hi = int(cfg.get("risk.score_max", 100)) if cfg else 100
    scaled = np.round(np.asarray(p_calibrated, dtype=np.float64) * (hi - lo) + lo)
    return np.clip(scaled, lo, hi).astype(np.int16)


@dataclass(frozen=True)
class TierPolicy:
    """Score bands and the action each one triggers."""

    medium_lo: int
    high_lo: int
    source: str          # "config" or "derived"
    actions: tuple[str, str, str] = ("auto_approve", "step_up_auth", "manual_review")

    def assign(self, scores: np.ndarray) -> np.ndarray:
        tier = np.full(len(scores), "low", dtype=object)
        tier[scores >= self.medium_lo] = "medium"
        tier[scores >= self.high_lo] = "high"
        return tier

    def action_for(self, tier: str) -> str:
        return {"low": self.actions[0], "medium": self.actions[1],
                "high": self.actions[2]}[tier]

    def to_dict(self) -> dict:
        return asdict(self)


def policy_from_config(cfg: Config) -> TierPolicy:
    tiers = cfg.require("risk.tiers")
    return TierPolicy(
        medium_lo=int(tiers["medium"]["lo"]),
        high_lo=int(tiers["high"]["lo"]),
        source="config",
        actions=(tiers["low"]["action"], tiers["medium"]["action"], tiers["high"]["action"]),
    )


def cost_optimal_probability(amounts: np.ndarray, cfg: Config) -> float:
    """The break-even probability implied by the cost model.

    Solving ``p * C_FN = (1 - p) * C_FP`` gives ``p* = C_FP / (C_FP + C_FN)``.
    C_FN varies per transaction, so the population-level threshold uses the
    mean amount; ``per_transaction_threshold`` gives the exact per-row version.
    """
    c_fp = float(cfg.require("risk.cost_fp_fixed"))
    multiplier = float(cfg.require("risk.cost_fn_multiplier"))
    c_fn = multiplier * float(np.mean(amounts))
    return c_fp / (c_fp + c_fn)


def per_transaction_threshold(amounts: np.ndarray, cfg: Config) -> np.ndarray:
    """Break-even probability for each transaction, given its own amount."""
    c_fp = float(cfg.require("risk.cost_fp_fixed"))
    c_fn = float(cfg.require("risk.cost_fn_multiplier")) * np.asarray(amounts, dtype=np.float64)
    return c_fp / (c_fp + np.maximum(c_fn, 1e-9))


def capacity_score_threshold(
    scores: np.ndarray, times: np.ndarray, cfg: Config
) -> int:
    """Lowest score whose alert volume fits the analyst budget.

    The budget is expressed as a fraction of daily transaction volume, so it
    scales with traffic rather than being a fixed count.
    """
    budget_pct = float(cfg.require("risk.daily_review_capacity_pct"))
    n_days = max(1.0, (float(times.max()) - float(times.min())) / SECONDS_PER_DAY)
    per_day = budget_pct * len(scores) / n_days
    allowed = max(1, min(int(round(per_day * n_days)), len(scores)))

    ordered = np.sort(scores)[::-1]
    threshold = int(ordered[allowed - 1])
    log.info(
        "review capacity: %.2f%% of volume = %.0f alerts/day over %.0f days "
        "-> %d alerts total, score >= %d",
        100 * budget_pct, per_day, n_days, allowed, threshold,
    )
    return threshold


def derive_policy(
    p_calibrated: np.ndarray,
    scores: np.ndarray,
    amounts: np.ndarray,
    times: np.ndarray,
    cfg: Config,
) -> TierPolicy:
    """Cost-and-capacity-derived tier boundaries.

    ``high_lo`` is the stricter of the cost break-even score and the capacity
    score - the operation cannot review more than its budget even when the
    economics say it should. ``medium_lo`` is set at a third of the break-even
    probability, the band where a step-up challenge is cheap enough to be worth
    issuing but a hold is not justified.
    """
    p_star = cost_optimal_probability(amounts, cfg)
    cost_score = int(np.clip(round(100 * p_star), 1, 100))
    cap_score = capacity_score_threshold(scores, times, cfg)

    high_lo = int(max(cost_score, cap_score))
    medium_lo = int(np.clip(round(100 * p_star / 3.0), 1, max(high_lo - 1, 1)))

    tiers = cfg.require("risk.tiers")
    log.info(
        "derived tiers: break-even p*=%.4f (score %d), capacity score %d "
        "-> medium>=%d, high>=%d",
        p_star, cost_score, cap_score, medium_lo, high_lo,
    )
    return TierPolicy(
        medium_lo=medium_lo,
        high_lo=high_lo,
        source="derived",
        actions=(tiers["low"]["action"], tiers["medium"]["action"], tiers["high"]["action"]),
    )


def expected_cost(
    y_true: np.ndarray, flagged: np.ndarray, amounts: np.ndarray, cfg: Config
) -> dict[str, float]:
    """Currency cost of a decision vector under the configured cost model."""
    c_fp = float(cfg.require("risk.cost_fp_fixed"))
    multiplier = float(cfg.require("risk.cost_fn_multiplier"))
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)

    missed = y_true & ~flagged
    false_alarm = ~y_true & flagged

    loss_fn = float(multiplier * np.sum(amounts[missed]))
    loss_fp = float(c_fp * np.sum(false_alarm))
    baseline = float(multiplier * np.sum(amounts[y_true]))  # flag nothing

    return {
        "missed_fraud_count": int(missed.sum()),
        "missed_fraud_value": round(loss_fn, 2),
        "false_alarm_count": int(false_alarm.sum()),
        "false_alarm_cost": round(loss_fp, 2),
        "total_cost": round(loss_fn + loss_fp, 2),
        "cost_if_nothing_flagged": round(baseline, 2),
        "cost_reduction_pct": round(100 * (1 - (loss_fn + loss_fp) / baseline), 2)
        if baseline > 0 else float("nan"),
        "alert_rate": round(float(flagged.mean()), 5),
    }


def tier_breakdown(
    tiers: np.ndarray, y_true: np.ndarray, amounts: np.ndarray, policy: TierPolicy
) -> pd.DataFrame:
    """Per-tier volume, capture and value - the table an operation reads."""
    rows = []
    total_fraud = int(np.sum(y_true))
    for tier in ("low", "medium", "high"):
        mask = tiers == tier
        n = int(mask.sum())
        caught = int(np.sum(y_true[mask]))
        rows.append(
            {
                "tier": tier,
                "action": policy.action_for(tier),
                "transactions": n,
                "share_of_volume": round(n / max(len(tiers), 1), 5),
                "fraud_in_tier": caught,
                "share_of_all_fraud": round(caught / max(total_fraud, 1), 5),
                "precision_in_tier": round(caught / n, 5) if n else float("nan"),
                "fraud_value_in_tier": round(float(np.sum(amounts[mask & (y_true == 1)])), 2),
            }
        )
    return pd.DataFrame(rows)


def decide(
    p_calibrated: np.ndarray,
    amounts: np.ndarray,
    times: np.ndarray,
    cfg: Config,
    policy: TierPolicy | None = None,
) -> pd.DataFrame:
    """Full decision output for a batch: score, tier, action, break-even flag."""
    scores = risk_score(p_calibrated, cfg)
    policy = policy or policy_from_config(cfg)
    tiers = policy.assign(scores)
    per_row_threshold = per_transaction_threshold(amounts, cfg)

    return pd.DataFrame(
        {
            "risk_score": scores,
            "p_calibrated": np.asarray(p_calibrated, dtype=np.float32),
            "tier": tiers,
            "action": [policy.action_for(t) for t in tiers],
            "cost_flag": (p_calibrated > per_row_threshold).astype(np.int8),
            "break_even_p": per_row_threshold.astype(np.float32),
        }
    )

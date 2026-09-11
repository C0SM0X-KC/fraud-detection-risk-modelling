"""Governance layer (L5): SHAP reason codes.

Report objective 5 and gap G4. Every high-risk alert carries the features that
drove it, expressed as a short ordered list a reviewer can read, so the adverse
action is auditable rather than asserted.

Attribution is taken from the strongest tree model rather than from the fused
score. The fusion step is a linear combination of four base scores; explaining
it in terms of "xgboost contributed 0.4" tells a reviewer nothing, whereas
"velocity on this card in the last hour: 11 transactions" is an actual reason.
The meta-learner weights are reported separately so the two levels are both
visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("frds.evaluation.explain")


def tree_shap_values(model, X: pd.DataFrame, background_size: int = 2000) -> np.ndarray:
    """SHAP values for a tree ensemble, as an (n_rows, n_features) array."""
    import shap

    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("estimator", model)

    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(X, check_additivity=False)

    # Binary classifiers return either one array or a list of two.
    if isinstance(values, list):
        values = values[1] if len(values) == 2 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, features, classes)
        values = values[..., -1]
    return values


def global_importance(shap_values: np.ndarray, feature_names: list[str], top_n: int = 20) -> pd.DataFrame:
    """Mean absolute attribution per feature - the global read."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def reason_codes(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    row_positions: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    """Per-alert reason codes: the top-k features pushing the score upward.

    Only positive contributions are listed. A reviewer asking "why was this
    held" needs the evidence *for* the hold; the features arguing against it
    are noise in that conversation.
    """
    names = np.array(X.columns)
    records = []

    for pos in row_positions:
        contributions = shap_values[pos]
        order = np.argsort(contributions)[::-1][:top_k]
        reasons = []
        for j in order:
            if contributions[j] <= 0:
                continue
            value = X.iloc[pos, j]
            value_str = f"{value:.4g}" if isinstance(value, (int, float, np.number)) and pd.notna(value) else str(value)
            reasons.append(f"{names[j]}={value_str} (+{contributions[j]:.4f})")
        records.append(
            {
                "row": int(pos),
                "n_reasons": len(reasons),
                "reason_codes": " | ".join(reasons) if reasons else "no positive contribution",
                "top_feature": names[order[0]] if len(order) else "",
                "total_positive_shap": float(contributions[contributions > 0].sum()),
            }
        )

    return pd.DataFrame(records)


def explain_alerts(
    model,
    X: pd.DataFrame,
    alert_positions: np.ndarray,
    background_size: int = 2000,
    top_k: int = 5,
    max_alerts: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per-alert reason codes, global importance) for a set of alerts.

    ``max_alerts`` caps the work: TreeSHAP is exact but not free, and a report
    does not need every one of ten thousand alerts explained to demonstrate
    that each one can be.
    """
    if alert_positions.size == 0:
        log.warning("no alerts to explain")
        return pd.DataFrame(), pd.DataFrame()

    selected = alert_positions[:max_alerts]
    subset = X.iloc[selected].reset_index(drop=True)
    log.info("computing SHAP attributions for %d alerts", len(subset))

    values = tree_shap_values(model, subset, background_size)
    local = reason_codes(values, subset, np.arange(len(subset)), top_k)
    local["source_row"] = selected
    glob = global_importance(values, list(subset.columns))
    return local, glob

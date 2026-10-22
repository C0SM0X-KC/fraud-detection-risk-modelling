"""Governance layer (L5): SHAP reason codes."""

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

    if type(estimator).__name__ == "CatBoostClassifier":
        # CatBoost's own TreeSHAP: exact, and it understands the categorical
        # columns it was trained with, which the generic explainer does not.
        from catboost import Pool

        cats = estimator.get_cat_feature_indices()
        frame = X.copy()
        if cats:
            cols = [frame.columns[i] for i in cats]
            frame[cols] = frame[cols].fillna(-1).astype(np.int64)
        values = estimator.get_feature_importance(Pool(frame, cat_features=cats),
                                                  type="ShapValues")
        return np.asarray(values)[:, :-1]

    explainer = shap.TreeExplainer(estimator)
    # Check additivity (base + contributions == model output); fall back only where float precision trips it.
    try:
        values = explainer.shap_values(X, check_additivity=True)
    except Exception as exc:  # pragma: no cover - model-dependent
        log.warning("SHAP additivity could not be confirmed (%s); explaining "
                    "without the check", str(exc)[:120])
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
    """Per-alert reason codes: the top-k features pushing the score upward."""
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
        # Only name a top reason if it actually pushes towards fraud.
        has_evidence = len(order) and contributions[order[0]] > 0
        records.append(
            {
                "row": int(pos),
                "n_reasons": len(reasons),
                "reason_codes": " | ".join(reasons) if reasons else "no positive contribution",
                "top_feature": names[order[0]] if has_evidence else "",
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
    """Return (per-alert reason codes, global importance) for a set of alerts."""
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


def faithfulness(
    predict,
    to_score,
    X: pd.DataFrame,
    shap_values: np.ndarray,
    reference: pd.Series,
    ks: tuple[int, ...] = (1, 3, 5),
    n_random: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Deletion test: do the stated reasons actually drive the score?"""
    rng = np.random.default_rng(seed)
    base = to_score(predict(X)).astype(np.float64)
    values = X.to_numpy(np.float64, copy=True)
    ref = reference.reindex(X.columns).to_numpy(np.float64)
    order = np.argsort(-shap_values, axis=1)
    n, m = values.shape
    rows = []

    def rescore(columns: np.ndarray) -> np.ndarray:
        modified = values.copy()
        idx = np.arange(n)[:, None]
        modified[idx, columns] = ref[columns]
        frame = pd.DataFrame(modified, columns=X.columns).astype(X.dtypes.to_dict(), errors="ignore")
        return to_score(predict(frame)).astype(np.float64)

    for k in ks:
        k = min(k, m)
        top = rescore(order[:, :k])
        random_scores = np.mean([
            rescore(np.stack([rng.choice(m, k, replace=False) for _ in range(n)]))
            for _ in range(n_random)
        ], axis=0)
        drop_top, drop_random = base - top, base - random_scores
        rows.append({
            "k": k,
            "alerts": n,
            "mean_score_before": round(float(base.mean()), 2),
            "mean_score_drop_top": round(float(drop_top.mean()), 2),
            "mean_score_drop_random": round(float(drop_random.mean()), 2),
            "drop_ratio_top_vs_random": round(
                float(drop_top.mean() / max(drop_random.mean(), 1e-6)), 1),
            "share_falling_below_half_top": round(float(np.mean(top < base / 2)), 3),
            "share_falling_below_half_random": round(float(np.mean(random_scores < base / 2)), 3),
        })
    return pd.DataFrame(rows)

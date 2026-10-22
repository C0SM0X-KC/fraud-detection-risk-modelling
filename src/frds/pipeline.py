"""Stage orchestration: the five workflow stages of the report, as functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import Config
from .data import load as data_load
from .data.split import chronological_split, load_split, save_split
from .evaluation import drift, explain, latency, metrics
from .features import online
from .features.build import build_features, load_features, save_features
from .fusion.calibration import (
    ProbabilityCalibrator,
    expected_calibration_error,
    reliability_curve,
)
from .fusion.risk import (
    derive_policy,
    expected_cost,
    policy_from_config,
    risk_score,
    tier_breakdown,
)
from .fusion.stacking import MeanLogitBlend, StackingMetaLearner, load_combiner
from .models.train import load_scores, save_scores, score_matrix, train_base_models
from .utils import get_logger, set_seed, timed, write_json

log = get_logger("frds.pipeline")


# ------------------------------------------------------------ stage 01 ----
def stage_prepare(cfg: Config) -> Path:
    set_seed(int(cfg.require("project.seed")))
    with timed("prepare", log):
        txn, ident = data_load.load_raw(cfg)
        merged = data_load.merge_tables(txn, ident, cfg)
        return data_load.save_prepared(merged, cfg)


# ------------------------------------------------------------ stage 02 ----
def stage_features(cfg: Config) -> Path:
    set_seed(int(cfg.require("project.seed")))
    with timed("features", log):
        df = data_load.load_prepared(cfg)
        split = chronological_split(df, cfg)
        save_split(split, cfg, df)

        X, y, spec, keys = build_features(df, cfg, split)
        # Keep the time column with the features for drift, latency and capacity checks.
        keys["__time"] = df[cfg.require("data.time_col")].to_numpy(np.int64)
        keys["__amount"] = df[cfg.require("data.amount_col")].to_numpy(np.float64)
        path = save_features(X, y, spec, keys, cfg)
        # The online scorer's entity history and fitted encoders, so a new
        # transaction can be featurised exactly as these rows were.
        online.save_state(df, X, y, spec, cfg)
        return path


# ------------------------------------------------------------ stage 03 ----
def stage_train(cfg: Config, force_refit: bool = False) -> Path:
    set_seed(int(cfg.require("project.seed")))
    with timed("train", log):
        X, y, categorical, keys = load_features(cfg)
        split = load_split(cfg)
        times = keys["__time"]
        entity_keys = {k: v for k, v in keys.items() if not k.startswith("__")}
        results, oof_positions, valid_positions = train_base_models(
            X, y, split, cfg, categorical, times, entity_keys, force_refit
        )
        return save_scores(results, oof_positions, split, y, cfg, valid_positions)


# ------------------------------------------------------------ stage 04 ----
def stage_fuse(cfg: Config) -> dict:
    """Fuse the base scores, then calibrate whichever score validation prefers."""
    set_seed(int(cfg.require("project.seed")))
    with timed("fuse", log):
        scores = load_scores(cfg)
        Z_oof, names = score_matrix(scores, "oof")
        Z_valid, _ = score_matrix(scores, "valid")
        Z_test, _ = score_matrix(scores, "test")
        y_valid = scores["y_valid"]

        meta = StackingMetaLearner(seed=int(cfg.require("project.seed")))
        meta.fit(Z_oof, scores["y_oof"], names)

        from sklearn.metrics import average_precision_score

        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "stacked": (meta.predict_proba(Z_valid), meta.predict_proba(Z_test)),
        }
        for i, name in enumerate(names):
            candidates[name] = (Z_valid[:, i], Z_test[:, i])

        # Equal-weight blends of the top tabular models, ranked on validation only.
        single_rank = {
            name: float(average_precision_score(y_valid, Z_valid[:, i]))
            for i, name in enumerate(names)
            if name not in {"lstm", "tcn", "hetero_gnn"}
        }
        ordered = sorted(single_rank, key=single_rank.get, reverse=True)
        blends: dict[str, MeanLogitBlend] = {}
        for size in (2, 3):
            if len(ordered) >= size:
                members = ordered[:size]
                blend = MeanLogitBlend(members)
                cols = [names.index(m) for m in members]
                key = f"blend_top{size}"
                blends[key] = blend
                candidates[key] = (blend.predict_proba(Z_valid[:, cols]),
                                   blend.predict_proba(Z_test[:, cols]))

        ranking = {
            name: float(average_precision_score(y_valid, valid_scores))
            for name, (valid_scores, _) in candidates.items()
        }
        selected = max(ranking, key=ranking.get)
        allow = bool(cfg.get("fusion.select_on_validation", True))
        if not allow:
            selected = "stacked"

        for name, value in sorted(ranking.items(), key=lambda kv: -kv[1]):
            log.info("  validation AUC-PR  %-22s %.4f%s", name, value,
                     "   <- selected" if name == selected else "")
        if selected != "stacked":
            log.warning(
                "stacking did not win on validation (%.4f vs %.4f for %s); "
                "the decision layer will use %s",
                ranking["stacked"], ranking[selected], selected, selected,
            )

        p_valid_raw, p_test_raw = candidates[selected]

        # Isotonic needs enough positives; fall back to Platt scaling below that.
        method = str(cfg.require("fusion.calibration"))
        positives = int(np.sum(y_valid))

        # Too few positives to calibrate at all: leave the score uncalibrated and warn.
        minimum = int(cfg.get("fusion.min_calibration_positives", 50))
        if positives < minimum:
            log.warning(
                "only %d positives in the calibration set (floor %d); skipping "
                "calibration - the score is reported uncalibrated",
                positives, minimum,
            )
            method = "none"

        floor = int(cfg.get("fusion.isotonic_min_positives", 500))
        if method == "isotonic" and positives < floor:
            log.warning(
                "only %d positives available for calibration (floor %d); "
                "using Platt scaling instead of isotonic",
                positives, floor,
            )
            method = "sigmoid"

        calibrator = ProbabilityCalibrator(method)
        calibrator.fit(p_valid_raw, y_valid)
        p_test = calibrator.transform(p_test_raw)

        artifacts = cfg.path("artifacts")
        meta.save(artifacts / "meta_learner.joblib")
        if selected in blends:
            blends[selected].save(artifacts / "blend.joblib")
        calibrator.save(artifacts / "calibrator.joblib")
        np.savez_compressed(
            artifacts / "fused_scores.npz",
            p_test_raw=p_test_raw,
            p_test_calibrated=p_test,
            p_valid_raw=p_valid_raw,
            p_valid_calibrated=calibrator.transform(p_valid_raw),
        )

        summary = {
            "selected_score": selected,
            "selected_on": "validation AUC-PR",
            "validation_ranking": ranking,
            "stacking_won": selected == "stacked",
            "blend_members": blends[selected].model_names if selected in blends else None,
            "meta_learner_weights": meta.weights(),
            "calibration_method": calibrator.method,
            "calibration_positives": positives,
            "ece_before": expected_calibration_error(scores["y_test"], p_test_raw),
            "ece_after": expected_calibration_error(scores["y_test"], p_test),
        }
        write_json(summary, cfg.path("reports") / "metrics" / "fusion_summary.json")
        log.info(
            "decision layer uses '%s'; expected calibration error on test: %.6f -> %.6f",
            selected, summary["ece_before"], summary["ece_after"],
        )
        from .tracking import log_stage

        log_stage(cfg, "fuse", {f"valid_auc_pr_{k}": v for k, v in ranking.items()}
                  | {"ece_after": summary["ece_after"]})
        return summary


# ------------------------------------------------------------ stage 05 ----
def stage_evaluate(cfg: Config, skip_shap: bool = False, skip_latency: bool = False) -> dict:
    set_seed(int(cfg.require("project.seed")))
    with timed("evaluate", log):
        X, y, categorical, keys = load_features(cfg)
        split = load_split(cfg)
        scores = load_scores(cfg)
        fused = np.load(cfg.path("artifacts") / "fused_scores.npz")

        y_test = scores["y_test"]
        test_pos = scores["test_positions"]
        times_test = keys["__time"][test_pos]
        amounts_test = keys["__amount"][test_pos]

        fpr_targets = list(cfg.require("evaluation.fpr_targets"))
        names = [str(n) for n in scores["model_names"]]

        # -- model comparison -----------------------------------------
        from .utils import read_json

        fusion_summary = read_json(
            cfg.path("reports") / "metrics" / "fusion_summary.json"
        )
        selected = fusion_summary.get("selected_score", "stacked")
        score_map = {n: scores[f"test__{n}"] for n in names}
        if selected not in names:            # the stack or a blend
            score_map[selected] = fused["p_test_raw"]
        score_map[f"decision layer ({selected}, calibrated)"] = fused["p_test_calibrated"]

        results = [
            metrics.evaluate(y_test, s, fpr_targets, label=name)
            for name, s in score_map.items()
        ]
        table = metrics.comparison_table(results)
        tables_dir = cfg.path("reports") / "tables"
        figures_dir = cfg.path("reports") / "figures"
        table.to_csv(tables_dir / "model_comparison.csv", index=False)
        write_json(results, cfg.path("reports") / "metrics" / "model_metrics.json")
        log.info("\n%s", table.to_string(index=False))

        plots.precision_recall_curves(y_test, score_map, figures_dir / "pr_curves.png")
        plots.roc_curves(y_test, score_map, figures_dir / "roc_curves.png")
        plots.model_comparison(table, figures_dir / "model_comparison.png")

        # -- calibration ------------------------------------------------
        p_test = fused["p_test_calibrated"]
        curve = reliability_curve(y_test, p_test)
        ece = expected_calibration_error(y_test, p_test)
        plots.reliability_diagram(curve, ece, figures_dir / "reliability.png")

        # -- risk tiers -------------------------------------------------
        risk = risk_score(p_test, cfg)
        fixed_policy = policy_from_config(cfg)
        derived_policy = derive_policy(p_test, risk, amounts_test, times_test, cfg)

        tier_tables = {}
        cost_tables = {}
        for policy in (fixed_policy, derived_policy):
            tiers = policy.assign(risk)
            breakdown = tier_breakdown(tiers, y_test, amounts_test, policy)
            breakdown.to_csv(tables_dir / f"risk_tiers_{policy.source}.csv", index=False)
            tier_tables[policy.source] = breakdown.to_dict(orient="records")
            flagged = tiers != "low"
            cost_tables[policy.source] = expected_cost(y_test, flagged, amounts_test, cfg)
            log.info("tier policy '%s'\n%s", policy.source, breakdown.to_string(index=False))

        plots.score_distribution(
            risk, y_test, derived_policy.medium_lo, derived_policy.high_lo,
            figures_dir / "score_distribution.png",
        )

        # -- drift ------------------------------------------------------
        drift_frame = drift.score_drift(p_test, times_test, cfg)
        drift_frame.to_csv(tables_dir / "psi_score_drift.csv", index=False)
        plots.psi_over_windows(
            drift_frame, float(cfg.require("evaluation.psi_alert_threshold")),
            figures_dir / "psi_drift.png",
        )
        feature_drift_frame = drift.feature_drift(X.iloc[test_pos], times_test, cfg)
        feature_drift_frame.to_csv(tables_dir / "psi_feature_drift.csv", index=False)
        prevalence_frame = drift.prevalence_drift(y_test, times_test, cfg)
        prevalence_frame.to_csv(tables_dir / "prevalence_drift.csv", index=False)

        summary: dict = {
            "test_span_days": [
                round(float(times_test.min()) / 86_400, 1),
                round(float(times_test.max()) / 86_400, 1),
            ],
            "model_comparison": table.to_dict(orient="records"),
            "calibration": {"ece": ece, "reliability": curve},
            "tier_policies": {
                "config": fixed_policy.to_dict(),
                "derived": derived_policy.to_dict(),
            },
            "tier_breakdown": tier_tables,
            "cost": cost_tables,
            "drift": {
                "score_psi": drift_frame.to_dict(orient="records"),
                "worst_feature_psi": feature_drift_frame.head(10).to_dict(orient="records"),
                "prevalence": prevalence_frame.to_dict(orient="records"),
            },
        }

        # -- explanation ------------------------------------------------
        if not skip_shap:
            summary["explanation"] = _explain_stage(cfg, X, test_pos, risk,
                                                    derived_policy, tables_dir, figures_dir)

        # -- serving cost -----------------------------------------------
        if not skip_latency:
            entity_keys = {k: v for k, v in keys.items() if not k.startswith("__")}
            summary["latency"] = _latency_stage(
                cfg, X, test_pos, names, tables_dir, entity_keys
            )
            summary["latency_product"] = _product_latency_stage(cfg, test_pos, tables_dir)

        write_json(summary, cfg.path("reports") / "metrics" / "evaluation_summary.json")
        from .tracking import log_stage

        decision_row = next((r for r in results if r["label"].startswith("decision layer")), {})
        log_stage(cfg, "evaluate",
                  {f"test_{k}": decision_row.get(k) for k in
                   ("auc_pr", "auc_roc", "f1_best", "mcc", "brier")} | {"test_ece": ece},
                  [tables_dir / "model_comparison.csv"])
        return summary


def _explain_stage(cfg, X, test_pos, risk, policy, tables_dir, figures_dir) -> dict:
    """SHAP reason codes for the alerts the derived policy would raise."""
    from .models.registry import build

    model_dir = cfg.path("artifacts") / "models"

    # Explain the selected score; a learned stack falls back to its strongest tree model.
    from .utils import read_json

    fusion_path = cfg.path("reports") / "metrics" / "fusion_summary.json"
    selected = read_json(fusion_path).get("selected_score") if fusion_path.exists() else None
    tree_models = ("xgboost", "lightgbm", "catboost", "random_forest")

    def load(n):
        return type(build(n, cfg)).load(model_dir / f"{n}.joblib")

    combiner = load_combiner(cfg.path("artifacts"), selected) if selected else None
    if (isinstance(combiner, MeanLogitBlend)
            and all(m in tree_models for m in combiner.model_names)):
        # A logit blend is explained exactly: its SHAP values are the mean of
        # its members' (see fusion.stacking.MeanLogitBlend).
        members = [load(m) for m in combiner.model_names]
        name, faithful = selected, True

        def shap_values(frame):
            return np.mean([explain.tree_shap_values(m.model, frame) for m in members], axis=0)

        def predict(frame):
            return combiner.predict_proba(np.column_stack([m.predict_proba(frame) for m in members]))
    else:
        if selected in tree_models and (model_dir / f"{selected}.joblib").exists():
            name, faithful = selected, True
        else:
            available = [n for n in tree_models if (model_dir / f"{n}.joblib").exists()]
            if not available:
                log.warning("no tree model saved; skipping SHAP")
                return {"skipped": "no tree model available"}
            name, faithful = available[0], False
            log.warning(
                "decision layer uses '%s', which has no single tree to attribute "
                "through; reason codes are taken from %s as a proxy", selected, name,
            )
        single = load(name)

        def shap_values(frame):
            return explain.tree_shap_values(single.model, frame)

        predict = single.predict_proba

    alerts = np.flatnonzero(risk >= policy.high_lo)
    if alerts.size == 0:
        log.warning("derived policy raised no high-risk alerts; explaining the top 200 scores")
        alerts = np.argsort(risk)[::-1][:200]

    X_test = X.iloc[test_pos].reset_index(drop=True)
    max_alerts = int(cfg.get("evaluation.shap_max_alerts", 500))
    selected_alerts = alerts[:max_alerts]
    subset = X_test.iloc[selected_alerts].reset_index(drop=True)
    log.info("computing SHAP attributions for %d alerts", len(subset))
    values = shap_values(subset)
    local = explain.reason_codes(values, subset, np.arange(len(subset)), top_k=5)
    local["source_row"] = selected_alerts
    glob = explain.global_importance(values, list(subset.columns))
    if local.empty:
        return {"skipped": "no alerts"}

    # Plain-language versions of the same reason codes, for the report and
    # for anyone reading the CSV without the feature dictionary.
    from .evaluation.reasons import explain_row, group_attribution
    from .features.online import load_state

    try:
        fitted = load_state(cfg)["fitted"]
        categories = {c: v["categories"] for c, v in fitted.get("categoricals", {}).items()}
    except (FileNotFoundError, RuntimeError):
        categories = None
    local["reason_text"] = [
        " | ".join(r["text"] for r in
                   explain_row(values[i], subset.iloc[i], None, categories, 3)["risk_factors"])
        for i in range(len(subset))
    ]

    local.to_csv(tables_dir / "alert_reason_codes.csv", index=False)
    glob.to_csv(tables_dir / "shap_global_importance.csv", index=False)
    plots.shap_importance(glob, figures_dir / "shap_importance.png")

    groups = group_attribution(values, list(subset.columns))
    groups.to_csv(tables_dir / "shap_group_importance.csv", index=False)
    plots.shap_groups(groups, figures_dir / "shap_groups.png")
    try:
        plots.shap_beeswarm(values, subset, figures_dir / "shap_beeswarm.png",
                            int(cfg.require("evaluation.shap_max_display")))
    except Exception as exc:  # plotting must never take the stage down
        log.warning("beeswarm plot skipped: %s", exc)

    # Deletion test. Reference values are medians over ordinary (non-alert)
    # test traffic, so "removing" a feature means making it unremarkable.
    calibrator = ProbabilityCalibrator.load(cfg.path("artifacts") / "calibrator.joblib")
    ordinary = np.setdiff1d(np.arange(len(X_test)), alerts)
    sample = np.random.default_rng(0).choice(ordinary, min(5_000, len(ordinary)), replace=False)
    reference = X_test.iloc[sample].median(numeric_only=True)
    n_faith = min(len(subset), int(cfg.get("evaluation.faithfulness_alerts", 300)))
    faith = explain.faithfulness(
        predict,
        lambda p: risk_score(calibrator.transform(p), cfg),
        subset.iloc[:n_faith], values[:n_faith], reference,
    )
    faith.to_csv(tables_dir / "explanation_faithfulness.csv", index=False)
    plots.faithfulness(faith, figures_dir / "explanation_faithfulness.png")
    log.info("explanation faithfulness\n%s", faith.to_string(index=False))

    log.info("reason codes written for %d alerts (model: %s)", len(local), name)
    return {
        "model": name,
        "explains_decision_model": faithful,
        "decision_model": selected,
        "output_space": "log-odds before calibration",
        "alerts_explained": int(len(local)),
        "top_features": glob.head(10).to_dict(orient="records"),
        "evidence_groups": groups.to_dict(orient="records"),
        "faithfulness": faith.to_dict(orient="records"),
        "sample_reason_codes": local.head(5).to_dict(orient="records"),
    }


def _latency_stage(cfg, X, test_pos, names, tables_dir, entity_keys) -> dict:
    """Time the full serving path: base models -> fusion -> calibration."""
    from .models.graph import HeteroGNNScorer
    from .models.registry import build
    from .models.sequence import SequenceScorer

    model_dir = cfg.path("artifacts") / "models"
    scorers, skipped = [], []
    for name in names:
        path = model_dir / f"{name}.joblib"
        if not path.exists():
            skipped.append(f"{name} (not saved)")
            continue
        try:
            scorer = type(build(name, cfg)).load(path)
        except (NotImplementedError, KeyError) as exc:
            skipped.append(f"{name} ({exc.__class__.__name__})")
            continue
        if isinstance(scorer, (SequenceScorer, HeteroGNNScorer)):
            scorer.restore_context(X, entity_keys)
        if isinstance(scorer, HeteroGNNScorer):
            scorer.warm_memory(int(test_pos.min()))
        scorers.append(scorer)

    if skipped:
        log.warning("latency excludes: %s", ", ".join(skipped))
    if len(scorers) != len(names):
        # The meta-learner expects one column per enabled model; timing a
        # partial stack would measure a scorer that does not exist.
        log.warning("not every tier reloaded; timing the base tiers only")

    meta = StackingMetaLearner.load(cfg.path("artifacts") / "meta_learner.joblib")
    calibrator = ProbabilityCalibrator.load(cfg.path("artifacts") / "calibrator.joblib")
    complete = len(scorers) == len(names)

    def base_scores(frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for scorer in scorers:
            if isinstance(scorer, HeteroGNNScorer):
                columns.append(scorer.predict_online(frame))
            else:
                columns.append(scorer.predict_proba(frame))
        return np.column_stack(columns)

    def score_fn(frame: pd.DataFrame) -> np.ndarray:
        base = base_scores(frame)
        if not complete:
            return base.mean(axis=1)
        return calibrator.transform(meta.predict_proba(base))

    X_test = X.iloc[test_pos]
    frame = latency.latency_report(
        score_fn, X_test, sample_size=int(cfg.require("evaluation.latency_sample_size"))
    )
    frame.to_csv(tables_dir / "latency.csv", index=False)
    log.info("\n%s", frame.to_string(index=False))
    return {
        "path_timed": "base models + stacking + calibration" if complete
                      else "base models only (a tier failed to reload)",
        "models_timed": [s.name for s in scorers],
        "models_excluded": skipped,
        "measurements": frame.to_dict(orient="records"),
    }


def _product_latency_stage(cfg, test_pos, tables_dir) -> dict:
    """Time the deployed product path on raw test-span transactions."""
    from .product import DecisionService

    try:
        merged = data_load.load_prepared(cfg)
        start = int(test_pos.min())
        store = online.OnlineFeatureStore(online.load_state(cfg), cfg, upto=start)
        service = DecisionService(cfg, store=store)
    except (NotImplementedError, FileNotFoundError, RuntimeError) as exc:
        log.warning("product latency skipped: %s", exc)
        return {"skipped": str(exc)}

    n_single = int(cfg.get("evaluation.latency_single_samples", 500))
    batch = int(cfg.get("evaluation.latency_batch_size", 256))
    count = 2 * n_single + 10 * batch
    target = cfg.require("data.target_col")
    raws = merged.iloc[start: start + count].drop(columns=[target]).to_dict("records")
    del merged

    target_ms = float(cfg.get("evaluation.latency_target_ms", 100))
    frame = latency.product_latency(service, raws, target_ms, n_single, batch)
    frame.to_csv(tables_dir / "latency_product.csv", index=False)
    log.info("\n%s", frame.to_string(index=False))
    return {"target_ms": target_ms, "measurements": frame.to_dict(orient="records")}


# ------------------------------------------------------------- all -------
def stage_all(
    cfg: Config,
    skip_shap: bool = False,
    skip_latency: bool = False,
    force_refit: bool = False,
) -> dict:
    stage_prepare(cfg)
    stage_features(cfg)
    stage_train(cfg, force_refit=force_refit)
    stage_fuse(cfg)
    return stage_evaluate(cfg, skip_shap=skip_shap, skip_latency=skip_latency)

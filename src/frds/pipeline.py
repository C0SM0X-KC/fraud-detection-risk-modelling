"""Stage orchestration: the five workflow stages of the report, as functions.

    prepare   stage 01, ingestion            -> data/interim/merged.parquet
    features  stage 02, feature engineering  -> data/processed/features.parquet
    train     stage 03, model layer          -> artifacts/base_scores.npz
    fuse      stage 04, fusion + risk        -> artifacts/meta.joblib, calibrator
    evaluate  stage 05, decision + governance-> reports/

Each stage reads what the previous one wrote, so any stage can be re-run alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .config import Config
from .data import load as data_load
from .data.split import chronological_split, load_split, save_split
from .evaluation import drift, explain, latency, metrics
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
from .fusion.stacking import StackingMetaLearner
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
        # The time column travels with the features: drift, latency and the
        # capacity threshold all need it, and re-reading the merged frame just
        # to recover one column is wasteful.
        keys["__time"] = df[cfg.require("data.time_col")].to_numpy(np.int64)
        keys["__amount"] = df[cfg.require("data.amount_col")].to_numpy(np.float64)
        return save_features(X, y, spec, keys, cfg)


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
    """Fuse the base scores, then calibrate whichever score validation prefers.

    Stacking is fitted as specified: a meta-learner over out-of-fold base
    scores. But it is not adopted unconditionally. The meta-learner is trained
    on scores produced by *fold* models, while at serving time it receives
    scores from the final models - and when ``split.refit_on_valid`` is on,
    those final models were additionally refitted on the validation span. The
    wider that gap, the less the learned weights transfer, and on this benchmark
    the stack can end up below its own strongest member.

    So the decision layer scores every candidate on the validation span and
    carries the winner forward. The test span plays no part in the choice. When
    the stack wins it is used; when a single model wins, that is reported
    plainly rather than dressed up as an ensemble.
    """
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

        # Isotonic regression is non-parametric and needs enough positives to
        # estimate a step function; below that it fits noise and can leave the
        # score worse calibrated than it found it. Platt scaling assumes a
        # sigmoid, which is a strong assumption but a far more stable one in
        # small samples. The method actually used is recorded in the summary.
        method = str(cfg.require("fusion.calibration"))
        positives = int(np.sum(y_valid))

        # Below a floor there is no way to estimate a calibration curve at all,
        # and a confidently wrong one is worse than none: it collapses the score
        # range and empties the risk tiers. Leaving the score uncalibrated with
        # a warning is the honest failure mode.
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
        if selected == "stacked":
            score_map["stacked"] = fused["p_test_raw"]
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

        write_json(summary, cfg.path("reports") / "metrics" / "evaluation_summary.json")
        return summary


def _explain_stage(cfg, X, test_pos, risk, policy, tables_dir, figures_dir) -> dict:
    """SHAP reason codes for the alerts the derived policy would raise."""
    from .models.registry import build

    model_dir = cfg.path("artifacts") / "models"

    # Explain the model that actually makes the decisions. The fusion stage may
    # select a single base model over the stack, and a reason code describing
    # any other model would be explaining a decision nobody made. Only when the
    # stack itself is selected - which has no single tree to attribute through -
    # do we fall back to the strongest available tree model, and say so.
    from .utils import read_json

    fusion_path = cfg.path("reports") / "metrics" / "fusion_summary.json"
    selected = read_json(fusion_path).get("selected_score") if fusion_path.exists() else None
    tree_models = ("xgboost", "lightgbm", "random_forest")

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

    scorer_cls = type(build(name, cfg))
    scorer = scorer_cls.load(model_dir / f"{name}.joblib")


    alerts = np.flatnonzero(risk >= policy.high_lo)
    if alerts.size == 0:
        log.warning("derived policy raised no high-risk alerts; explaining the top 200 scores")
        alerts = np.argsort(risk)[::-1][:200]

    X_test = X.iloc[test_pos].reset_index(drop=True)
    local, glob = explain.explain_alerts(
        scorer.model,
        X_test,
        alerts,
        background_size=int(cfg.require("evaluation.shap_background_size")),
        top_k=5,
        max_alerts=500,
    )
    if local.empty:
        return {"skipped": "no alerts"}

    local.to_csv(tables_dir / "alert_reason_codes.csv", index=False)
    glob.to_csv(tables_dir / "shap_global_importance.csv", index=False)
    plots.shap_importance(glob, figures_dir / "shap_importance.png")
    log.info("reason codes written for %d alerts (model: %s)", len(local), name)
    return {
        "model": name,
        "explains_decision_model": faithful,
        "decision_model": selected,
        "output_space": "log-odds before calibration",
        "alerts_explained": int(len(local)),
        "top_features": glob.head(10).to_dict(orient="records"),
        "sample_reason_codes": local.head(5).to_dict(orient="records"),
    }


def _latency_stage(cfg, X, test_pos, names, tables_dir, entity_keys) -> dict:
    """Time the full serving path: base models -> fusion -> calibration.

    Every enabled tier is reloaded from disk and reconnected to the feature
    frame, so the number reported is the cost of the whole deployed scorer,
    not of the cheap half of it. The graph tier is warmed up to the start of
    the test span first and then scored online - a live system maintains its
    entity memories as traffic passes, it does not replay history per request.
    """
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

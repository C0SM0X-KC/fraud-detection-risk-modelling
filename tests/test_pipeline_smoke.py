"""End-to-end smoke test on a small synthetic fixture."""

from __future__ import annotations

import json

import numpy as np
import pytest

from frds import pipeline
from frds.config import load_config
from frds.data.synthetic import write_fixture


@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("frds_smoke")
    cfg = load_config()
    for kind in ("raw", "interim", "processed", "artifacts", "reports"):
        cfg.override(f"paths.{kind}", str(root / kind))
    cfg.ensure_dirs()
    for sub in ("figures", "tables", "metrics"):
        (root / "reports" / sub).mkdir(parents=True, exist_ok=True)

    # Small, fast, and only the cheap tiers: this test checks plumbing, not
    # model quality.
    cfg.override("models.enabled", ["logistic_regression", "lightgbm"])
    cfg.override("split.n_oof_folds", 2)
    cfg.override("features.max_v_columns", 20)
    cfg.override("evaluation.psi_windows", 3)
    cfg.override("evaluation.latency_sample_size", 200)

    # Big enough that boosting fits something (at 6,000 rows it stops at iteration 1).
    write_fixture(cfg.path("raw"), n_rows=40_000, n_cards=3_000,
                  n_devices=1_200, seed=1)

    pipeline.stage_prepare(cfg)
    pipeline.stage_features(cfg)
    pipeline.stage_train(cfg)
    fusion = pipeline.stage_fuse(cfg)
    summary = pipeline.stage_evaluate(cfg, skip_shap=False, skip_latency=True)
    return cfg, fusion, summary


def test_every_stage_writes_its_artifact(small_run):
    cfg, _, _ = small_run
    assert (cfg.path("interim") / "merged.parquet").exists()
    assert (cfg.path("interim") / "split_index.npz").exists()
    assert (cfg.path("processed") / "features.parquet").exists()
    assert (cfg.path("processed") / "feature_spec.json").exists()
    assert (cfg.path("artifacts") / "base_scores.npz").exists()
    assert (cfg.path("artifacts") / "meta_learner.joblib").exists()
    assert (cfg.path("artifacts") / "calibrator.joblib").exists()
    assert (cfg.path("artifacts") / "fused_scores.npz").exists()


def test_feature_spec_contains_every_group(small_run):
    cfg, _, _ = small_run
    spec = json.loads((cfg.path("processed") / "feature_spec.json").read_text())
    for group in ("base", "velocity", "recency_rolling", "graph", "mismatch",
                  "encoded_frequency", "encoded_label", "raw_numeric"):
        assert spec["groups"][group] > 0, f"feature group '{group}' is empty"
    assert spec["n_features"] == len(spec["columns"])


def test_split_summary_is_chronological(small_run):
    cfg, _, _ = small_run
    summary = json.loads(
        (cfg.path("reports") / "metrics" / "split_summary.json").read_text()
    )
    assert summary["train"]["t_end_day"] <= summary["valid"]["t_start_day"]
    assert summary["valid"]["t_end_day"] <= summary["test"]["t_start_day"]
    assert summary["strategy"] == "chronological"


def test_calibration_improves_the_expected_calibration_error(small_run):
    _, fusion, _ = small_run
    assert fusion["ece_after"] <= fusion["ece_before"] + 1e-9


def test_models_beat_the_no_skill_floor(small_run):
    _, _, summary = small_run
    comparison = summary["model_comparison"]
    assert len(comparison) >= 3      # base models plus the two fused rows
    best = max(row["auc_pr"] for row in comparison)
    assert best > 0.05, "every model is at or below the no-skill floor"
    assert all(row["auc_pr_lift_over_prevalence"] > 1.0 for row in comparison)


def test_risk_tiers_are_ordered_by_precision(small_run):
    _, _, summary = small_run
    for source in ("config", "derived"):
        rows = {r["tier"]: r for r in summary["tier_breakdown"][source]}
        assert sum(r["transactions"] for r in rows.values()) > 0
        # Tiers can legitimately be empty; compare only populated ones.
        populated = [
            r for tier in ("low", "medium", "high")
            if (r := rows[tier])["transactions"] > 0
        ]
        precisions = [r["precision_in_tier"] for r in populated]
        assert precisions == sorted(precisions), (
            f"{source} tiers are not ordered by precision: {precisions}"
        )


def test_cost_model_beats_flagging_nothing(small_run):
    _, _, summary = small_run
    cost = summary["cost"]["derived"]
    assert cost["total_cost"] < cost["cost_if_nothing_flagged"]
    assert 0.0 <= cost["alert_rate"] <= 1.0


def test_drift_report_has_one_row_per_window(small_run):
    cfg, _, summary = small_run
    windows = summary["drift"]["score_psi"]
    assert len(windows) == cfg.require("evaluation.psi_windows")
    assert windows[0]["status"] == "reference"
    assert all(np.isfinite(w["psi_vs_window0"]) for w in windows)


def test_alerts_carry_reason_codes(small_run):
    cfg, _, summary = small_run
    explanation = summary.get("explanation", {})
    if explanation.get("skipped"):
        pytest.skip(explanation["skipped"])
    assert explanation["alerts_explained"] > 0
    assert (cfg.path("reports") / "tables" / "alert_reason_codes.csv").exists()
    sample = explanation["sample_reason_codes"][0]
    assert sample["reason_codes"]
    assert sample["n_reasons"] >= 0


def test_figures_are_produced(small_run):
    cfg, _, _ = small_run
    figures = cfg.path("reports") / "figures"
    for name in ("pr_curves.png", "roc_curves.png", "reliability.png",
                 "score_distribution.png", "psi_drift.png", "model_comparison.png"):
        assert (figures / name).exists(), f"missing figure {name}"

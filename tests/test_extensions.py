"""The Phase-1 items added after the core pipeline: fusion blends, tuning,
robustness, the BAF transfer check, streaming, and the explanation quality
checks. Each is run end to end on small synthetic data - these tests check
contracts and wiring, not model quality.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from frds import pipeline, stream, transfer
from frds.config import load_config
from frds.data.synthetic import write_fixture
from frds.evaluation import reasons, robustness
from frds.fusion.stacking import MeanLogitBlend
from frds.models import tuning


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("frds_ext")
    cfg = load_config()
    for kind in ("raw", "interim", "processed", "artifacts", "reports"):
        cfg.override(f"paths.{kind}", str(root / kind))
    cfg.ensure_dirs()
    for sub in ("figures", "tables", "metrics"):
        (root / "reports" / sub).mkdir(parents=True, exist_ok=True)
    cfg.override("models.enabled", ["logistic_regression", "xgboost", "lightgbm"])
    cfg.override("models.xgboost.n_estimators", 150)
    cfg.override("models.lightgbm.n_estimators", 150)
    cfg.override("split.n_oof_folds", 2)
    cfg.override("features.max_v_columns", 20)
    cfg.override("evaluation.psi_windows", 3)
    cfg.override("evaluation.latency_single_samples", 40)
    cfg.override("evaluation.latency_batch_size", 32)
    cfg.override("evaluation.latency_sample_size", 200)
    write_fixture(cfg.path("raw"), n_rows=30_000, n_cards=2_500, n_devices=1_000, seed=11)
    pipeline.stage_prepare(cfg)
    pipeline.stage_features(cfg)
    pipeline.stage_train(cfg)
    fusion = pipeline.stage_fuse(cfg)
    summary = pipeline.stage_evaluate(cfg)
    return cfg, fusion, summary


def test_fusion_considers_blends(trained):
    _, fusion, _ = trained
    assert {"blend_top2", "blend_top3"} <= set(fusion["validation_ranking"])


def test_blend_is_the_mean_of_logits():
    blend = MeanLogitBlend(["a", "b"])
    p = blend.predict_proba(np.array([[0.5, 0.5], [0.9, 0.1]]))
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0.5)


def test_explanation_outputs_are_written(trained):
    cfg, _, summary = trained
    tables = cfg.path("reports") / "tables"
    groups = pd.read_csv(tables / "shap_group_importance.csv")
    assert groups["share"].sum() == pytest.approx(1.0, abs=1e-6)
    codes = pd.read_csv(tables / "alert_reason_codes.csv")
    assert "reason_text" in codes and codes["reason_text"].astype(str).str.len().gt(0).any()


def test_reasons_are_faithful(trained):
    cfg, _, _ = trained
    faith = pd.read_csv(cfg.path("reports") / "tables" / "explanation_faithfulness.csv")
    top3 = faith[faith["k"] == 3].iloc[0]
    # Removing the stated top reasons must move the score more than removing
    # the same number of arbitrary features.
    assert top3["mean_score_drop_top"] > top3["mean_score_drop_random"]


def test_product_latency_is_measured_against_the_target(trained):
    cfg, _, summary = trained
    rows = {r["mode"]: r for r in summary["latency_product"]["measurements"]}
    assert "single_transaction" in rows and "single_transaction_with_reasons" in rows
    assert rows["single_transaction"]["target_ms"] == cfg.require("evaluation.latency_target_ms")
    assert rows["single_transaction"]["p50_ms"] > 0


def test_every_feature_has_a_readable_reason(trained):
    cfg, _, _ = trained
    spec = json.loads((cfg.path("processed") / "feature_spec.json").read_text())
    for name in spec["columns"]:
        text = reasons.describe(name, 1.0)
        assert text and name not in ("",)
        assert reasons.feature_group(name) in reasons.GROUPS


def test_robustness_attack_beats_noise(trained):
    cfg, _, _ = trained
    cfg.override("robustness.epsilons", [0.5, 1.0])
    cfg.override("robustness.max_attacked", 200)
    cfg.override("robustness.pgd_steps", 15)
    frame = robustness.run(cfg)
    assert set(frame["attack"]) == {"pgd", "random"}
    assert frame["evasion_rate"].between(0, 1).all()
    pgd = frame[frame["attack"] == "pgd"]["evasion_rate"].to_numpy()
    rnd = frame[frame["attack"] == "random"]["evasion_rate"].to_numpy()
    assert pgd.sum() >= rnd.sum()
    assert (cfg.path("reports") / "figures" / "robustness.png").exists()


def test_tuning_writes_tuned_parameters(trained):
    cfg, _, _ = trained
    results = tuning.tune(cfg, ["xgboost"], n_trials=3, timeout=300)
    assert results["xgboost"]["trials"] >= 1
    stored = json.loads((cfg.path("artifacts") / "tuned_params.json").read_text())
    assert "max_depth" in stored["xgboost"]["params"]
    cfg.override("tuning.use_tuned", True)
    from frds.models.registry import model_params

    assert model_params("xgboost", cfg)["max_depth"] == stored["xgboost"]["params"]["max_depth"]
    cfg.override("tuning.use_tuned", False)


def test_streaming_memory_transport_scores_everything(trained):
    cfg, _, _ = trained
    summary = stream.run(cfg, transport="memory", rows=300, batch_size=32)
    assert summary["transactions"] == 300
    assert sum(summary["tiers"].values()) == 300
    assert summary["end_to_end_p95_ms"] > 0


def _fake_baf(path, n=12_000, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "fraud_bool": (rng.random(n) < 0.03).astype(int),
        "income": rng.choice([0.1, 0.5, 0.9], n),
        "name_email_similarity": rng.random(n),
        "customer_age": rng.choice([20, 30, 40, 50, 60], n),
        "velocity_6h": rng.gamma(2, 2000, n),
        "credit_risk_score": rng.normal(130, 60, n),
        "payment_type": rng.choice(["AA", "AB", "AC"], n),
        "employment_status": rng.choice(["CA", "CB"], n),
        "housing_status": rng.choice(["BA", "BB", "BC"], n),
        "source": rng.choice(["INTERNET", "TELEAPP"], n),
        "device_os": rng.choice(["linux", "windows", "other"], n),
        "month": rng.integers(0, 8, n),
    })
    signal = frame["fraud_bool"] == 1
    frame.loc[signal, "credit_risk_score"] += 80
    frame.loc[signal, "name_email_similarity"] *= 0.4
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_baf_transfer_runs_the_same_protocol(trained):
    cfg, _, _ = trained
    _fake_baf(cfg.path("raw") / cfg.require("transfer.baf_file"))
    table = transfer.run(cfg)
    assert table["label"].str.startswith("decision layer").any()
    assert (table["auc_pr"] > 0.03).all()
    meta = json.loads((cfg.path("reports") / "metrics" / "baf_transfer.json").read_text())
    assert meta["rows"]["test"] > 0
    assert 0 <= meta["fairness"]["fpr_ratio_min_over_max"] <= 1

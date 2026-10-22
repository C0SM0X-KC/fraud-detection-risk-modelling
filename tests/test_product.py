"""The product path: raw transaction in, decision and reasons out."""

from __future__ import annotations

import numpy as np
import pytest

from frds import pipeline
from frds.config import load_config
from frds.data import load as data_load
from frds.data.split import load_split
from frds.data.synthetic import write_fixture
from frds.features.build import load_features
from frds.features.online import OnlineFeatureStore
from frds.product import DecisionService, VERDICTS


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("frds_product")
    cfg = load_config()
    for kind in ("raw", "interim", "processed", "artifacts", "reports"):
        cfg.override(f"paths.{kind}", str(root / kind))
    cfg.ensure_dirs()
    for sub in ("figures", "tables", "metrics"):
        (root / "reports" / sub).mkdir(parents=True, exist_ok=True)
    cfg.override("models.enabled", ["xgboost"])
    cfg.override("models.xgboost.n_estimators", 200)
    cfg.override("split.n_oof_folds", 2)
    cfg.override("features.max_v_columns", 20)
    write_fixture(cfg.path("raw"), n_rows=30_000, n_cards=2_500, n_devices=1_000, seed=5)
    pipeline.stage_prepare(cfg)
    pipeline.stage_features(cfg)
    pipeline.stage_train(cfg)
    pipeline.stage_fuse(cfg)
    merged = data_load.load_prepared(cfg)
    X, y, _, _ = load_features(cfg)
    return cfg, merged, X, y.to_numpy(), load_split(cfg)


def test_fast_path_matches_predict_proba(trained):
    cfg, merged, X, y, split = trained
    service = DecisionService(cfg)
    model = service.models[0]
    rows = X.iloc[split.test[:300]]
    np.testing.assert_allclose(model.predict(rows), model.scorer.predict_proba(rows),
                               rtol=1e-5, atol=1e-6)


def test_reason_codes_add_up_to_the_model_output(trained):
    cfg, merged, X, y, split = trained
    service = DecisionService(cfg)
    model = service.explainer
    rows = X.iloc[split.test[:200]]
    values, base = model.contributions(rows)
    margin = values.sum(axis=1) + base
    p = 1.0 / (1.0 + np.exp(-margin))
    np.testing.assert_allclose(p, model.predict(rows), rtol=1e-4, atol=1e-5)


def test_online_score_equals_offline_score(trained):
    cfg, merged, X, y, split = trained
    start = int(split.test[0])
    store = OnlineFeatureStore.from_disk(cfg, upto=start)
    service = DecisionService(cfg, store=store)
    offline = service._probability(X.iloc[start:start + 50])[1]
    online = []
    for position in range(start, start + 50):
        decision = service.score(merged.iloc[position], explain=False)
        store.record_outcome(decision["transaction_id"], int(y[position]))
        online.append(decision["probability"])
    np.testing.assert_allclose(online, offline, rtol=1e-4, atol=1e-6)


def test_a_decision_carries_score_tier_verdict_and_reasons(trained):
    cfg, merged, X, y, split = trained
    service = DecisionService(cfg)
    raw = merged.iloc[int(split.test[-1])].to_dict()
    raw.pop(cfg.require("data.id_col"))
    raw.pop(cfg.require("data.target_col"))
    decision = service.score(raw)

    assert 0 <= decision["risk_score"] <= 100
    assert decision["tier"] in VERDICTS
    assert decision["verdict"] == VERDICTS[decision["tier"]][0]
    reasons = decision["reasons"]
    assert reasons["risk_factors"] or reasons["mitigating_factors"]
    for factor in reasons["risk_factors"]:
        assert factor["contribution"] > 0 and factor["text"]
    assert {"features_ms", "model_ms", "decision_ms", "explanation_ms",
            "total_ms"} <= set(decision["timings_ms"])


def test_micro_batch_matches_single_scoring(trained):
    cfg, merged, X, y, split = trained
    raws = [merged.iloc[int(p)].to_dict() for p in split.test[-20:]]
    for r in raws:
        # Re-sent as new arrivals: the store scores forward in time only, so
        # each takes the next timestamp and a fresh id.
        for key in ("data.target_col", "data.time_col", "data.id_col"):
            r.pop(cfg.require(key))
    single = DecisionService(cfg)
    batch = DecisionService(cfg)
    one_by_one = [single.score(r, explain=False)["probability"] for r in raws]
    together = [d["probability"] for d in batch.score_many(raws)]
    np.testing.assert_allclose(one_by_one, together, rtol=1e-6)


def test_feedback_moves_the_next_score(trained):
    cfg, merged, X, y, split = trained
    service = DecisionService(cfg)
    raw = merged.iloc[int(split.test[-1])].to_dict()
    for key in (cfg.require("data.id_col"), cfg.require("data.target_col")):
        raw.pop(key)
    raw["card1"] = 77_777_777
    first = service.score(raw)
    before = service.score(raw, remember=False, explain=False)["raw_model_score"]
    assert service.record_outcome(first["transaction_id"], True)
    after = service.score(raw, remember=False, explain=False)["raw_model_score"]
    assert after != before

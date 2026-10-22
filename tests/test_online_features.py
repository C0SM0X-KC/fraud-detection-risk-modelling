"""The online feature store must reproduce the offline feature frame exactly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frds import pipeline
from frds.config import load_config
from frds.data import load as data_load
from frds.data.split import load_split
from frds.data.synthetic import write_fixture
from frds.features.build import load_features
from frds.features.online import OnlineFeatureStore

REPLAYED = 400


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("frds_online")
    cfg = load_config()
    for kind in ("raw", "interim", "processed", "artifacts", "reports"):
        cfg.override(f"paths.{kind}", str(root / kind))
    cfg.ensure_dirs()
    (root / "reports" / "metrics").mkdir(parents=True, exist_ok=True)
    cfg.override("features.max_v_columns", 20)
    write_fixture(cfg.path("raw"), n_rows=20_000, n_cards=1_500, n_devices=600, seed=3)
    pipeline.stage_prepare(cfg)
    pipeline.stage_features(cfg)

    merged = data_load.load_prepared(cfg)
    X, y, _, _ = load_features(cfg)
    split = load_split(cfg)
    return cfg, merged, X, y.to_numpy(), split


def _assert_rows_equal(online: pd.DataFrame, offline: pd.Series, position: int) -> None:
    bad = []
    for col in offline.index:
        a = float(online[col].iloc[0])
        b = float(offline[col])
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a, b, rtol=1e-4, atol=1e-4):
            bad.append(f"{col}: online={a!r} offline={b!r}")
    assert not bad, f"row {position} differs in {len(bad)} features:\n" + "\n".join(bad[:15])


def test_columns_and_dtypes_match(prepared):
    cfg, merged, X, y, split = prepared
    store = OnlineFeatureStore.from_disk(cfg, upto=int(split.test[0]))
    row = store.features(merged.iloc[int(split.test[0])])
    assert list(row.columns) == list(X.columns)
    assert dict(row.dtypes.astype(str)) == dict(X.dtypes.astype(str))


def test_replayed_test_span_matches_offline_features(prepared):
    cfg, merged, X, y, split = prepared
    start = int(split.test[0])
    store = OnlineFeatureStore.from_disk(cfg, upto=start)
    for position in range(start, start + REPLAYED):
        features, _ = store.score_and_update(merged.iloc[position], label=int(y[position]))
        _assert_rows_equal(features, X.iloc[position], position)


def test_a_burst_on_one_card_raises_its_velocity(prepared):
    cfg, merged, X, y, split = prepared
    store = OnlineFeatureStore.from_disk(cfg)
    template = merged.iloc[int(split.test[-1])].to_dict()
    template.pop(cfg.require("data.id_col"))

    first = store.features(template)
    for _ in range(3):
        store.update(template)
    after = store.features(template)
    assert after["vel_card_cnt_1h"].iloc[0] == first["vel_card_cnt_1h"].iloc[0] + 3
    assert after["recency_card_seconds"].iloc[0] <= 1.0


def test_an_unseen_card_starts_from_the_prior(prepared):
    cfg, merged, X, y, split = prepared
    store = OnlineFeatureStore.from_disk(cfg)
    raw = merged.iloc[int(split.test[-1])].to_dict()
    raw.pop(cfg.require("data.id_col"))
    raw["card1"] = 99_999_999
    raw["addr1"] = 999.0
    features = store.features(raw)
    assert features["vel_card_cnt_24h"].iloc[0] == 0
    assert np.isnan(features["recency_card_seconds"].iloc[0])
    expected = store.global_rate
    assert features["graph_card_prior_fraud_rate"].iloc[0] == pytest.approx(expected, rel=1e-5)


def test_a_confirmed_outcome_feeds_the_prior_fraud_rate(prepared):
    cfg, merged, X, y, split = prepared
    store = OnlineFeatureStore.from_disk(cfg)
    raw = merged.iloc[int(split.test[-1])].to_dict()
    raw.pop(cfg.require("data.id_col"))
    raw["card1"] = 88_888_888

    txn_id = store.update(raw)                    # outcome unknown on arrival
    unknown = store.features(raw)["graph_card_prior_fraud_rate"].iloc[0]
    assert store.record_outcome(txn_id, 1)        # analyst confirms fraud
    confirmed = store.features(raw)["graph_card_prior_fraud_rate"].iloc[0]
    assert confirmed > unknown


def test_time_cannot_run_backwards(prepared):
    cfg, merged, X, y, split = prepared
    store = OnlineFeatureStore.from_disk(cfg)
    raw = merged.iloc[int(split.train[0])].to_dict()
    raw.pop(cfg.require("data.id_col"))
    with pytest.raises(ValueError, match="precedes"):
        store.features(raw)

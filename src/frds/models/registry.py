"""Model registry: config name -> scorer instance."""

from __future__ import annotations

import json

from ..config import Config
from .base import BaseScorer
from .baselines import LogisticRegressionScorer, RandomForestScorer
from .boosting import CatBoostScorer, LightGBMScorer, XGBoostScorer
from .foundation import TabPFNScorer
from .graph import HeteroGNNScorer
from .sequence import LSTMScorer, TCNScorer

_REGISTRY: dict[str, type[BaseScorer]] = {
    # baseline tier
    LogisticRegressionScorer.name: LogisticRegressionScorer,
    RandomForestScorer.name: RandomForestScorer,
    # tabular tier
    XGBoostScorer.name: XGBoostScorer,
    LightGBMScorer.name: LightGBMScorer,
    CatBoostScorer.name: CatBoostScorer,
    # sequential tier
    LSTMScorer.name: LSTMScorer,
    TCNScorer.name: TCNScorer,
    # relational tier
    HeteroGNNScorer.name: HeteroGNNScorer,
    # foundation tier
    TabPFNScorer.name: TabPFNScorer,
}

#: Which tier of report Table 4.2 each scorer belongs to.
TIERS: dict[str, str] = {
    "logistic_regression": "baseline",
    "random_forest": "baseline",
    "xgboost": "tabular",
    "lightgbm": "tabular",
    "catboost": "tabular",
    "lstm": "sequential",
    "tcn": "sequential",
    "hetero_gnn": "relational",
    "tabpfn": "foundation",
}


def register(scorer_cls: type[BaseScorer]) -> type[BaseScorer]:
    _REGISTRY[scorer_cls.name] = scorer_cls
    return scorer_cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def model_params(name: str, cfg: Config) -> dict:
    """Hyper-parameters for one model: the config, plus tuned values if enabled."""
    params = dict(cfg.get(f"models.{name}", {}) or {})
    if cfg.get("tuning.use_tuned", False):
        path = cfg.path("artifacts") / "tuned_params.json"
        if path.exists():
            tuned = json.loads(path.read_text(encoding="utf-8")).get(name, {})
            params.update(tuned.get("params", {}))
    return params


def build(name: str, cfg: Config) -> BaseScorer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {available()}")
    return _REGISTRY[name](model_params(name, cfg))


def build_enabled(cfg: Config) -> list[BaseScorer]:
    names = cfg.require("models.enabled")
    return [build(name, cfg) for name in names]

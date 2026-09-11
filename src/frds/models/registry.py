"""Model registry: config name -> scorer instance.

Adding the sequence, graph and foundation tiers means registering them here;
nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from ..config import Config
from .base import BaseScorer
from .baselines import LogisticRegressionScorer, RandomForestScorer
from .boosting import LightGBMScorer, XGBoostScorer
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


def build(name: str, cfg: Config) -> BaseScorer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Available: {available()}")
    params = cfg.get(f"models.{name}", {}) or {}
    return _REGISTRY[name](params)


def build_enabled(cfg: Config) -> list[BaseScorer]:
    names = cfg.require("models.enabled")
    return [build(name, cfg) for name in names]

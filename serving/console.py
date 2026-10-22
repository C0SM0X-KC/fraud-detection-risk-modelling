"""Shared resources for the console pages, loaded once per server process."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "serving"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from frds.config import load_config  # noqa: E402
from frds.product import DecisionService  # noqa: E402
from scorer import FraudScorer  # noqa: E402

#: The queue speaks in verbs an analyst uses, and the same word is used from
#: tier to action to column heading so the vocabulary is learnable.
TIER_VERB = {"low": "Clear", "medium": "Challenge", "high": "Hold"}


@st.cache_resource(show_spinner="Loading the replay harness...")
def get_scorer() -> FraudScorer:
    return FraudScorer()


@st.cache_resource(show_spinner="Loading the decision service and entity history...")
def get_service() -> DecisionService:
    return DecisionService(load_config())


@st.cache_data
def reference_scores() -> np.ndarray | None:
    """Calibrated scores on the calibration holdout: the drift reference."""
    path = load_config().path("artifacts") / "fused_scores.npz"
    if not path.exists():
        return None
    return np.load(path)["p_valid_calibrated"]


def reports_dir() -> Path:
    return load_config().path("reports")


def read_table(name: str) -> pd.DataFrame | None:
    path = reports_dir() / "tables" / name
    return pd.read_csv(path) if path.exists() else None


def figure(name: str) -> Path | None:
    path = reports_dir() / "figures" / name
    return path if path.exists() else None

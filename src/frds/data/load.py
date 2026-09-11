"""Ingestion layer (L1): read the raw tables, merge, validate, downcast.

Corresponds to stage 01 of the workflow figure and layer L1 of the
architecture figure: schema validation, de-duplication and chronological
ordering by TransactionDT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, human_bytes

log = get_logger("frds.data.load")


class SchemaError(RuntimeError):
    """Raised when the raw tables do not match the expected contract."""


def reduce_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Downcast numeric columns in place-ish. 434 float64 columns will not fit
    comfortably in a laptop's RAM; float32 is ample for tree models."""
    before = df.memory_usage(deep=True).sum()
    for col in df.columns:
        kind = df[col].dtype.kind
        if kind == "f":
            df[col] = df[col].astype(np.float32)
        elif kind in "iu":
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
    after = df.memory_usage(deep=True).sum()
    if verbose:
        log.info(
            "memory %s -> %s (%.0f%% saved)",
            human_bytes(before), human_bytes(after),
            100 * (1 - after / max(before, 1)),
        )
    return df


def load_raw(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train_transaction.csv and train_identity.csv."""
    raw = cfg.path("raw")
    txn_path = raw / cfg.require("data.transaction_file")
    id_path = raw / cfg.require("data.identity_file")

    for p in (txn_path, id_path):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run 'python -m frds acquire' (real data) or "
                f"'python -m frds synth' (fixture)."
            )

    log.info("reading %s", txn_path.name)
    txn = pd.read_csv(txn_path, low_memory=False)
    log.info("reading %s", id_path.name)
    ident = pd.read_csv(id_path, low_memory=False)
    log.info("transaction %s, identity %s", txn.shape, ident.shape)
    return txn, ident


def _normalise_identity_columns(ident: pd.DataFrame) -> pd.DataFrame:
    """Kaggle ships train_identity with ``id_01`` and test_identity with
    ``id-01``. Normalise to the underscore form so downstream code is stable."""
    renames = {c: c.replace("id-", "id_") for c in ident.columns if c.startswith("id-")}
    return ident.rename(columns=renames) if renames else ident


def validate_schema(txn: pd.DataFrame, cfg: Config) -> None:
    id_col = cfg.require("data.id_col")
    time_col = cfg.require("data.time_col")
    target = cfg.require("data.target_col")
    amount = cfg.require("data.amount_col")

    missing = [c for c in (id_col, time_col, target, amount) if c not in txn.columns]
    if missing:
        raise SchemaError(f"transaction table is missing required columns: {missing}")

    if txn[target].dropna().isin([0, 1]).all() is np.False_:
        raise SchemaError(f"{target} must be binary 0/1")
    if txn[time_col].isna().any():
        raise SchemaError(f"{time_col} contains nulls; chronological order is undefined")

    prevalence = float(txn[target].mean())
    if not 0.0 < prevalence < 0.5:
        raise SchemaError(f"implausible fraud prevalence: {prevalence:.4f}")
    log.info(
        "schema ok | rows=%d cols=%d prevalence=%.3f%% span=%.1f days",
        len(txn), txn.shape[1], 100 * prevalence,
        (txn[time_col].max() - txn[time_col].min()) / 86_400,
    )


def merge_tables(txn: pd.DataFrame, ident: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Left-join identity onto transaction, de-duplicate, sort chronologically."""
    id_col = cfg.require("data.id_col")
    time_col = cfg.require("data.time_col")

    ident = _normalise_identity_columns(ident)
    validate_schema(txn, cfg)

    before = len(txn)
    txn = txn.drop_duplicates(subset=[id_col], keep="first")
    if len(txn) != before:
        log.warning("dropped %d duplicate TransactionID rows", before - len(txn))

    df = txn.merge(ident, on=id_col, how="left")
    coverage = df["DeviceInfo"].notna().mean() if "DeviceInfo" in df else float("nan")
    log.info("identity coverage after merge: %.1f%%", 100 * coverage)

    # Ties on TransactionDT are broken by TransactionID, which is issued in
    # arrival order. Without a stable total order the split is not reproducible.
    df = df.sort_values([time_col, id_col], kind="mergesort").reset_index(drop=True)

    sample = cfg.get("data.sample_rows")
    if sample:
        df = df.iloc[: int(sample)].copy()
        log.info("development sample: first %d rows of the timeline", len(df))

    return reduce_memory(df)


def load_prepared(cfg: Config) -> pd.DataFrame:
    """Read the merged parquet written by the ``prepare`` stage."""
    path = cfg.path("interim") / "merged.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m frds prepare' first.")
    df = pd.read_parquet(path)
    log.info("loaded merged frame %s from %s", df.shape, path.name)
    return df


def save_prepared(df: pd.DataFrame, cfg: Config) -> Path:
    path = cfg.path("interim") / "merged.parquet"
    df.to_parquet(path, index=False)
    log.info("wrote %s (%s)", path.name, human_bytes(path.stat().st_size))
    return path

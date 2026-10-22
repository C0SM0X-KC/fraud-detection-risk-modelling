"""Schema-compatible synthetic stand-in for IEEE-CIS Fraud Detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import get_logger

log = get_logger("frds.data.synthetic")

PRODUCT_CODES = np.array(["W", "C", "R", "H", "S"])
EMAIL_DOMAINS = np.array(
    ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com",
     "outlook.com", "comcast.net", "icloud.com", "mail.com", "protonmail.com"]
)
DEVICE_TYPES = np.array(["desktop", "mobile"])
DEVICE_MODELS = np.array(
    ["Windows", "iOS Device", "MacOS", "Trident/7.0", "SM-G950U Build/R16NW",
     "rv:11.0", "Moto G (5) Build/NPPS25", "SM-J700M Build/MMB29K", "Linux"]
)

SECONDS_PER_DAY = 86_400


def generate(
    n_rows: int = 60_000,
    fraud_rate: float = 0.035,
    span_days: int = 182,
    n_cards: int = 6_000,
    n_devices: int = 2_500,
    n_ring_devices: int = 40,
    label_noise: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (transaction_df, identity_df) mirroring the IEEE-CIS schema."""
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n_rows * fraud_rate))
    n_legit = n_rows - n_fraud

    # ---- population --------------------------------------------------
    card_ids = rng.integers(1000, 1000 + n_cards, size=n_cards)
    device_ids = np.arange(n_devices)
    ring_devices = rng.choice(device_ids, size=n_ring_devices, replace=False)
    # More compromised cards than ring devices, so the shared-device edge is
    # the only thing tying those cards together.
    ring_cards = rng.choice(card_ids, size=max(n_cards // 12, 2), replace=False)

    # ---- timeline ----------------------------------------------------
    span_seconds = span_days * SECONDS_PER_DAY
    legit_dt = np.sort(rng.integers(0, span_seconds, size=n_legit))

    # Fraud arrives in bursts: pick anchors, then cluster around them.
    n_bursts = max(1, n_fraud // 6)
    anchors = rng.integers(0, span_seconds, size=n_bursts)
    burst_of = rng.integers(0, n_bursts, size=n_fraud)
    fraud_dt = np.clip(
        anchors[burst_of] + rng.exponential(1800, size=n_fraud).astype(np.int64),
        0,
        span_seconds - 1,
    )

    dt = np.concatenate([legit_dt, fraud_dt])
    is_fraud = np.concatenate([np.zeros(n_legit, np.int8), np.ones(n_fraud, np.int8)])
    order = np.argsort(dt, kind="stable")
    dt, is_fraud = dt[order], is_fraud[order]
    n = len(dt)
    fraud_mask = is_fraud == 1
    n_fr = int(fraud_mask.sum())
    n_ok = n - n_fr
    day = dt // SECONDS_PER_DAY

    # ---- entity assignment -------------------------------------------
    card1 = np.empty(n, dtype=np.int64)
    card1[~fraud_mask] = rng.choice(card_ids, size=n_ok)
    from_ring = rng.random(n_fr) < 0.70
    card1[fraud_mask] = np.where(
        from_ring,
        rng.choice(ring_cards, size=n_fr),
        rng.choice(card_ids, size=n_fr),
    )

    device = np.empty(n, dtype=np.int64)
    device[~fraud_mask] = rng.choice(device_ids, size=n_ok)
    # Ring share drifts upward over the span - this is the concept drift.
    ring_pressure = 0.45 + 0.35 * (day[fraud_mask] / max(span_days - 1, 1))
    on_ring_device = rng.random(n_fr) < ring_pressure
    device[fraud_mask] = np.where(
        on_ring_device,
        rng.choice(ring_devices, size=n_fr),
        rng.choice(device_ids, size=n_fr),
    )

    # ---- amount -------------------------------------------------------
    amt = np.exp(rng.normal(3.9, 0.95, size=n))
    drift = 1.0 + 0.25 * (day / max(span_days - 1, 1))
    amt[fraud_mask] *= rng.uniform(0.9, 2.1, size=n_fr) * drift[fraud_mask]
    amt = np.round(np.clip(amt, 0.25, 32_000.0), 3)

    # ---- categorical fields -------------------------------------------
    product = np.empty(n, dtype=object)
    product[~fraud_mask] = rng.choice(
        PRODUCT_CODES, size=n_ok, p=[0.74, 0.11, 0.07, 0.05, 0.03]
    )
    product[fraud_mask] = rng.choice(
        PRODUCT_CODES, size=n_fr, p=[0.58, 0.20, 0.11, 0.07, 0.04]
    )

    p_email = np.empty(n, dtype=object)
    p_email[~fraud_mask] = rng.choice(
        EMAIL_DOMAINS, size=n_ok,
        p=[0.42, 0.16, 0.11, 0.09, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02],
    )
    p_email[fraud_mask] = rng.choice(
        EMAIL_DOMAINS, size=n_fr,
        p=[0.33, 0.14, 0.10, 0.15, 0.06, 0.05, 0.04, 0.04, 0.05, 0.04],
    )
    # R_emaildomain usually matches P_emaildomain for legitimate traffic.
    mismatch = rng.random(n) < np.where(fraud_mask, 0.26, 0.09)
    r_email = np.where(mismatch, rng.choice(EMAIL_DOMAINS, size=n), p_email)
    r_email = np.where(rng.random(n) < 0.60, None, r_email)  # ~60% missing

    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n, dtype=np.int64),
            "isFraud": is_fraud,
            "TransactionDT": dt.astype(np.int64) + SECONDS_PER_DAY,
            "TransactionAmt": amt,
            "ProductCD": product,
            "card1": card1,
            "card2": rng.integers(100, 600, size=n),
            "card3": np.where(rng.random(n) < 0.88, 150, 185),
            "card4": rng.choice(
                ["visa", "mastercard", "american express", "discover"],
                size=n, p=[0.65, 0.30, 0.03, 0.02],
            ),
            "card5": rng.integers(100, 240, size=n),
            "card6": rng.choice(["debit", "credit"], size=n, p=[0.74, 0.26]),
            "addr1": rng.integers(100, 540, size=n).astype(float),
            "addr2": np.full(n, 87.0),
            "dist1": np.where(rng.random(n) < 0.60, np.nan,
                              rng.integers(0, 3000, size=n)),
            "dist2": np.where(rng.random(n) < 0.93, np.nan,
                              rng.integers(0, 6000, size=n)),
            "P_emaildomain": p_email,
            "R_emaildomain": r_email,
        }
    )

    # ---- C1..C14 counting features ------------------------------------
    # Only a few carry weak signal; shifting all would make the task trivially separable.
    informative_c = {1, 2, 4, 11, 13}
    for i in range(1, 15):
        base = rng.poisson(1.6, size=n).astype(float)
        if i in informative_c:
            base[fraud_mask] += rng.poisson(0.30, size=n_fr)
        frame[f"C{i}"] = base

    # ---- D1..D15 timedelta features -----------------------------------
    # D1 = days since the card was first seen, consistent with `day` so day - D1 works as a uid.
    card_origin = pd.Series(day).groupby(pd.Series(card1)).transform("min").to_numpy()
    frame["D1"] = (day - card_origin).astype(float)
    for i in range(2, 16):
        col = rng.exponential(30 + 4 * i, size=n)
        col[rng.random(n) < (0.20 + 0.03 * i)] = np.nan
        frame[f"D{i}"] = np.round(col, 1)

    # ---- M1..M9 match flags -------------------------------------------
    for i in range(1, 10):
        if i == 4:
            out = rng.choice(["M0", "M1", "M2"], size=n).astype(object)
        else:
            p_true = np.where(fraud_mask, 0.74, 0.82)
            out = np.where(rng.random(n) < p_true, "T", "F").astype(object)
        out[rng.random(n) < 0.35] = None
        frame[f"M{i}"] = out

    # ---- V1..V60 Vesta engineered features -----------------------------
    # Blocks share a missingness pattern, as they do in the real table.
    signal = fraud_mask.astype(float)
    # Only 8 of 60 carry a weak shift; more would make the classes perfectly separable.
    informative_v = set(rng.choice(np.arange(1, 61), size=8, replace=False).tolist())
    for block_start in range(1, 61, 12):
        block_missing = rng.random(n) < rng.uniform(0.05, 0.55)
        for j in range(block_start, min(block_start + 12, 61)):
            weight = rng.uniform(-0.30, 0.30) if j in informative_v else 0.0
            col = rng.normal(0, 1, size=n) + weight * signal
            col[block_missing] = np.nan
            frame[f"V{j}"] = np.round(col, 4)

    # ---- identity table -------------------------------------------------
    # Identity covers about a quarter of rows, more often for fraud.
    has_identity = rng.random(n) < np.where(fraud_mask, 0.34, 0.23)
    idx = np.flatnonzero(has_identity)
    m = idx.size
    identity = pd.DataFrame({"TransactionID": frame["TransactionID"].to_numpy()[idx]})
    for i in range(1, 12):
        identity[f"id_{i:02d}"] = np.round(rng.normal(0, 1, size=m) * 10, 1)
    categorical_ids = {12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38}
    for i in range(12, 39):
        if i in categorical_ids:
            identity[f"id_{i:02d}"] = rng.choice(
                ["Found", "NotFound", "New", "Unknown"], size=m,
                p=[0.45, 0.35, 0.12, 0.08],
            )
        else:
            identity[f"id_{i:02d}"] = np.round(rng.normal(0, 1, size=m), 3)
    identity["DeviceType"] = rng.choice(DEVICE_TYPES, size=m)
    # DeviceInfo carries the ring signal: shared strings across unrelated cards.
    dev_of_row = device[idx]
    identity["DeviceInfo"] = [
        f"{DEVICE_MODELS[d % len(DEVICE_MODELS)]}-{d:05d}" for d in dev_of_row
    ]

    # Label noise (unreported fraud, non-fraud disputes), equal both ways to keep the 3.5% prior.
    n_flip = int(round(label_noise * n_fr * 2))
    if n_flip:
        pos_idx = np.flatnonzero(fraud_mask)
        neg_idx = np.flatnonzero(~fraud_mask)
        k = min(n_flip, len(pos_idx), len(neg_idx))
        frame.loc[rng.choice(pos_idx, size=k, replace=False), "isFraud"] = 0
        frame.loc[rng.choice(neg_idx, size=k, replace=False), "isFraud"] = 1

    log.info(
        "synthetic fixture: %d rows, %.2f%% fraud, %d days, %d identity rows, "
        "%d labels flipped",
        n, 100 * float(frame["isFraud"].mean()), span_days, m, 2 * n_flip // 2,
    )
    return frame, identity


def write_fixture(raw_dir: Path, **kwargs) -> tuple[Path, Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    txn, ident = generate(**kwargs)
    txn_path = raw_dir / "train_transaction.csv"
    id_path = raw_dir / "train_identity.csv"
    txn.to_csv(txn_path, index=False)
    ident.to_csv(id_path, index=False)
    (raw_dir / "SYNTHETIC_FIXTURE").write_text(
        "These CSVs were generated by frds.data.synthetic. They are a wiring\n"
        "fixture, not the IEEE-CIS benchmark. Delete this marker and the CSVs\n"
        "before loading the real data.\n",
        encoding="utf-8",
    )
    log.info("wrote %s and %s", txn_path.name, id_path.name)
    return txn_path, id_path


def is_fixture(raw_dir: Path) -> bool:
    return (Path(raw_dir) / "SYNTHETIC_FIXTURE").exists()

"""Replay the start of the real test span through the online store and compare.

    python scripts/check_online_parity.py [rows]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from frds.config import load_config
from frds.data import load as data_load
from frds.data.split import load_split
from frds.features.build import load_features
from frds.features.online import OnlineFeatureStore


def main(rows: int) -> None:
    cfg = load_config()
    merged = data_load.load_prepared(cfg)
    X, y, _, _ = load_features(cfg)
    start = int(load_split(cfg).test[0])
    store = OnlineFeatureStore.from_disk(cfg, upto=start)
    y_arr = y.to_numpy()

    mismatched_rows, column_misses, times = 0, {}, []
    for position in range(start, start + rows):
        t0 = time.perf_counter()
        online, _ = store.score_and_update(merged.iloc[position], label=int(y_arr[position]))
        times.append((time.perf_counter() - t0) * 1000)
        a = online.iloc[0].to_numpy(np.float64)
        b = X.iloc[position].to_numpy(np.float64)
        same = (np.isnan(a) & np.isnan(b)) | np.isclose(a, b, rtol=1e-4, atol=1e-4)
        if not same.all():
            mismatched_rows += 1
            for col in X.columns[~same]:
                column_misses[col] = column_misses.get(col, 0) + 1

    print(f"rows replayed: {rows}")
    print(f"rows with every one of {X.shape[1]} features equal: {rows - mismatched_rows} "
          f"({100 * (rows - mismatched_rows) / rows:.2f}%)")
    for col, n in sorted(column_misses.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  differs: {col} on {n} rows")
    print(f"online featurisation: p50 {np.percentile(times, 50):.2f} ms, "
          f"p95 {np.percentile(times, 95):.2f} ms")

    agg = [c for c in X.columns if c.startswith("agg_") and c.endswith("_mean")]
    filled = X[agg].notna().mean().sort_values()
    print("non-missing share of per-entity means (after the NaN fix):")
    for col, share in filled.head(8).items():
        print(f"  {col}: {share:.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)

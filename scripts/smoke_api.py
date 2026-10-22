"""Exercise the running product API end to end, as a client would.

    docker compose up -d api
    python scripts/smoke_api.py [http://localhost:8000]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def call(method: str, path: str, body: dict | None = None, timeout: float = 300) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def show(decision: dict, label: str) -> None:
    t = decision["timings_ms"]
    print(f"{label}: risk {decision['risk_score']:>3}  {decision['verdict']:<13} -> "
          f"{decision['action']:<24} p={decision['probability']:.4f}  "
          f"{t['total_ms']:.1f} ms (features {t['features_ms']:.1f}, model {t['model_ms']:.2f})")


def main() -> int:
    for _ in range(60):
        try:
            health = call("GET", "/health", timeout=600)
            break
        except Exception:  # noqa: BLE001 - server still starting
            time.sleep(5)
    else:
        print("API did not come up")
        return 1
    print("health:", {k: health[k] for k in ("model", "calibration", "tier_boundaries", "history_rows")})

    example = call("GET", "/transactions/example?fraud=true")
    txn = example["transaction"]
    print(f"example: a real transaction that was fraud, {len(txn)} fields, amount {txn['TransactionAmt']}")

    first = call("POST", "/transactions/score", {"transaction": txn})
    show(first, "first     ")
    for r in first["reasons"]["risk_factors"][:3]:
        print(f"    + {r['text']}  ({r['group']}, +{r['contribution']:.3f})")
    for r in first["reasons"]["mitigating_factors"][:2]:
        print(f"    - {r['text']}  ({r['group']}, {r['contribution']:.3f})")

    again = dict(txn)
    again2 = call("POST", "/transactions/score", {"transaction": again, "explain": False})
    show(again2, "same card ")

    outcome = call("POST", f"/transactions/{first['transaction_id']}/outcome", {"is_fraud": True})
    print("outcome recorded:", outcome)
    after = call("POST", "/transactions/score", {"transaction": again, "explain": False})
    show(after, "after fb  ")

    legit = call("GET", "/transactions/example?fraud=false")["transaction"]
    show(call("POST", "/transactions/score", {"transaction": legit, "explain": False}), "legit ex  ")

    batch = call("POST", "/transactions/score-batch",
                 {"transactions": [legit] * 32, "remember": False})
    print(f"batch of {batch['count']}: {batch['ms_per_transaction']:.2f} ms per transaction")
    return 0


if __name__ == "__main__":
    sys.exit(main())

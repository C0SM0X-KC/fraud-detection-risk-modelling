"""HTTP scoring endpoint over the same decision layer the dashboard uses.

    uvicorn serving.api:app --port 8000
    curl "http://localhost:8000/score/472432"

Report Table 4.6 lists a scoring endpoint, so here is one. It is deliberately
thin: every decision it makes comes from ``FraudScorer``, the same object the
replay console drives, so the two cannot disagree about what the system does.

A caveat this endpoint does not hide. Scoring a single transaction per request
costs around 126 ms at the median, against 0.24 ms amortised in batches of a
thousand - almost all of it per-call overhead in the scientific Python stack
rather than model complexity. Every response carries its own service time, and
``/score/batch`` exists because batching is what actually meets the latency
requirement. A deployment would micro-batch arrivals or export the model to a
compiled runtime; this endpoint is honest about being the slow path.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "serving"))

from scorer import FraudScorer  # noqa: E402

app = FastAPI(
    title="Fraud detection and risk modelling",
    description="Calibrated risk scoring with cost-sensitive tiers and reason codes.",
    version="1.0.0",
)

_scorer: FraudScorer | None = None


def scorer() -> FraudScorer:
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer()
    return _scorer


class ScoreResponse(BaseModel):
    position: int
    risk_score: int = Field(description="Bounded 0-100 risk score")
    probability: float = Field(description="Calibrated probability of fraud")
    tier: str
    action: str
    amount: float
    day: float
    service_ms: float = Field(description="Server-side time for this request")


class BatchRequest(BaseModel):
    positions: list[int] = Field(
        description="Row positions in the feature frame, in chronological order",
        max_length=4096,
    )
    explain: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **scorer().describe()}


@app.get("/score/{position}", response_model=ScoreResponse)
def score_one(position: int) -> ScoreResponse:
    svc = scorer()
    if not 0 <= position < len(svc.X):
        raise HTTPException(404, f"position {position} is outside the frame")

    started = time.perf_counter()
    row = svc.score_positions(np.array([position])).iloc[0]
    elapsed = (time.perf_counter() - started) * 1000.0

    return ScoreResponse(
        position=int(row.position),
        risk_score=int(row.risk_score),
        probability=float(row.p_calibrated),
        tier=str(row.tier),
        action=str(row.action),
        amount=float(row.amount),
        day=float(row.day),
        service_ms=round(elapsed, 2),
    )


@app.post("/score/batch")
def score_batch(request: BatchRequest) -> dict:
    """Score many positions at once - the path that meets the latency budget."""
    svc = scorer()
    positions = np.asarray(request.positions, dtype=int)
    if positions.size == 0:
        raise HTTPException(400, "no positions supplied")
    if positions.min() < 0 or positions.max() >= len(svc.X):
        raise HTTPException(404, "one or more positions are outside the frame")

    started = time.perf_counter()
    decisions = svc.score_positions(positions)
    elapsed = (time.perf_counter() - started) * 1000.0

    payload = decisions[
        ["position", "risk_score", "p_calibrated", "tier", "action", "amount", "day"]
    ].rename(columns={"p_calibrated": "probability"}).to_dict(orient="records")

    if request.explain:
        codes = svc.explain_positions(positions[:64])
        by_position = dict(zip(codes["position"], codes["reason_codes"])) if not codes.empty else {}
        for item in payload:
            item["reason_codes"] = by_position.get(item["position"])

    return {
        "count": len(payload),
        "service_ms": round(elapsed, 2),
        "ms_per_transaction": round(elapsed / len(payload), 4),
        "decisions": payload,
    }


@app.get("/explain/{position}")
def explain(position: int) -> dict:
    svc = scorer()
    codes = svc.explain_positions(np.array([position]))
    if codes.empty:
        raise HTTPException(503, "no tree model available for attribution")
    row = codes.iloc[0]
    return {
        "position": position,
        "top_feature": row["top_feature"],
        "reason_codes": str(row["reason_codes"]).split(" | "),
    }

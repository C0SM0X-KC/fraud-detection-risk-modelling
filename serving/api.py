"""HTTP scoring endpoint: the product, over HTTP."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "serving"))

from frds.config import load_config  # noqa: E402
from frds.product import DecisionService  # noqa: E402
from scorer import FraudScorer  # noqa: E402

app = FastAPI(
    title="Fraud detection and risk modelling",
    description="Calibrated 0-100 risk scoring with cost-sensitive tiers and "
                "plain-language reason codes.",
    version="2.0.0",
)

_scorer: FraudScorer | None = None
_service: DecisionService | None = None


def scorer() -> FraudScorer:
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer()
    return _scorer


def service() -> DecisionService:
    global _service
    if _service is None:
        _service = DecisionService(load_config())
    return _service


# --------------------------------------------------------------- product ----
class TransactionRequest(BaseModel):
    transaction: dict[str, Any] = Field(
        description="Raw IEEE-CIS fields, e.g. TransactionAmt, card1, addr1, "
                    "P_emaildomain, DeviceInfo. Omitted fields are treated as "
                    "missing. TransactionDT defaults to just after the latest "
                    "transaction seen.",
        examples=[{"TransactionAmt": 950.0, "ProductCD": "W", "card1": 13926,
                   "card4": "visa", "card6": "debit", "addr1": 315.0,
                   "P_emaildomain": "gmail.com"}],
    )
    remember: bool = Field(True, description="Append to the entity history")
    explain: bool = Field(True, description="Attach reason codes")
    top_k: int = Field(5, ge=1, le=20)


class BatchTransactionRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(max_length=4096)
    remember: bool = True


class OutcomeRequest(BaseModel):
    is_fraud: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **service().describe()}


@app.post("/transactions/score")
def score_transaction(request: TransactionRequest) -> dict:
    """Score one new transaction: risk score, verdict, action and reasons."""
    try:
        return service().score(request.transaction, remember=request.remember,
                               explain=request.explain, top_k=request.top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/transactions/score-batch")
def score_transactions(request: BatchTransactionRequest) -> dict:
    """Micro-batch: featurised in order, scored in one model call."""
    if not request.transactions:
        raise HTTPException(400, "no transactions supplied")
    started = time.perf_counter()
    try:
        decisions = service().score_many(request.transactions, remember=request.remember)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "count": len(decisions),
        "service_ms": round(elapsed, 2),
        "ms_per_transaction": round(elapsed / len(decisions), 4),
        "decisions": decisions,
    }


@app.post("/transactions/{transaction_id}/outcome")
def record_outcome(transaction_id: int, request: OutcomeRequest) -> dict:
    """Analyst feedback: the confirmed outcome joins the entity's fraud history."""
    if not service().record_outcome(transaction_id, request.is_fraud):
        raise HTTPException(404, f"transaction {transaction_id} was not scored by this service")
    return {"transaction_id": transaction_id, "recorded": True, "is_fraud": request.is_fraud}


@app.get("/transactions/example")
def example(fraud: bool = Query(False, description="Return an example that was fraud")) -> dict:
    """A real transaction from the end of the timeline, ready to edit and score."""
    pool = [t for t in service().templates() if t["was_fraud"] == fraud]
    if not pool:
        raise HTTPException(404, "no examples stored; re-run 'python -m frds features'")
    pick = pool[np.random.default_rng().integers(len(pool))]
    fields = {k: v for k, v in pick["fields"].items() if v is not None}
    return {"was_fraud": pick["was_fraud"], "transaction": fields}


# ------------------------------------------------------ evaluation harness --
class BatchRequest(BaseModel):
    positions: list[int] = Field(
        description="Row positions in the feature frame, in chronological order",
        max_length=4096,
    )
    explain: bool = False


@app.get("/score/{position}")
def score_one(position: int) -> dict:
    svc = scorer()
    if not 0 <= position < len(svc.X):
        raise HTTPException(404, f"position {position} is outside the frame")
    started = time.perf_counter()
    row = svc.score_positions(np.array([position])).iloc[0]
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "position": int(row.position),
        "risk_score": int(row.risk_score),
        "probability": float(row.p_calibrated),
        "tier": str(row.tier),
        "action": str(row.action),
        "amount": float(row.amount),
        "day": float(row.day),
        "service_ms": round(elapsed, 2),
    }


@app.post("/score/batch")
def score_batch(request: BatchRequest) -> dict:
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
        for item in payload[:64]:
            item["reasons"] = svc.explain_position(item["position"])
    return {
        "count": len(payload),
        "service_ms": round(elapsed, 2),
        "ms_per_transaction": round(elapsed / len(payload), 4),
        "decisions": payload,
    }


@app.get("/explain/{position}")
def explain(position: int) -> dict:
    svc = scorer()
    if not 0 <= position < len(svc.X):
        raise HTTPException(404, f"position {position} is outside the frame")
    reasons = svc.explain_position(position)
    if reasons is None:
        raise HTTPException(503, "no tree model available for attribution")
    return {"position": position, **reasons}

"""Streaming transport: transactions in on one topic, decisions out on another."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .data import load as data_load
from .data.split import load_split
from .features.online import OnlineFeatureStore, load_state
from .product import DecisionService
from .utils import get_logger, write_json

log = get_logger("frds.stream")

TRANSACTIONS = "transactions"
DECISIONS = "decisions"


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def encode(record: dict) -> bytes:
    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in record.items()}
    return json.dumps(clean, default=_json_default).encode("utf-8")


# ---------------------------------------------------------------- transports --
@dataclass
class MemoryTransport:
    """Two in-process queues standing in for the two topics."""

    topics: dict[str, queue.Queue] = field(
        default_factory=lambda: {TRANSACTIONS: queue.Queue(), DECISIONS: queue.Queue()})

    def send(self, topic: str, payload: bytes) -> None:
        self.topics[topic].put(payload)

    def poll(self, topic: str, max_records: int, timeout_s: float) -> list[bytes]:
        out = []
        deadline = time.perf_counter() + timeout_s
        while len(out) < max_records:
            remaining = deadline - time.perf_counter()
            try:
                out.append(self.topics[topic].get(timeout=max(remaining, 0.0)) if not out
                           else self.topics[topic].get_nowait())
            except queue.Empty:
                break
        return out

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class KafkaTransport:
    """kafka-python producer and consumer on a real broker."""

    def __init__(self, bootstrap: str, group: str = "frds-scorer") -> None:
        from kafka import KafkaConsumer, KafkaProducer

        self.producer = KafkaProducer(bootstrap_servers=bootstrap, linger_ms=1, acks=1)
        self.consumer = KafkaConsumer(
            TRANSACTIONS, bootstrap_servers=bootstrap, group_id=group,
            auto_offset_reset="earliest", enable_auto_commit=True,
        )

    def send(self, topic: str, payload: bytes) -> None:
        self.producer.send(topic, payload)

    def poll(self, topic: str, max_records: int, timeout_s: float) -> list[bytes]:
        batches = self.consumer.poll(timeout_ms=int(timeout_s * 1000), max_records=max_records)
        return [m.value for records in batches.values() for m in records]

    def flush(self) -> None:
        self.producer.flush()

    def close(self) -> None:
        self.producer.close()
        self.consumer.close()


# ------------------------------------------------------------------- run ----
def run(cfg: Config, transport: str = "memory", bootstrap: str = "localhost:9092",
        rows: int = 20_000, rate: float = 0.0, batch_size: int = 128,
        max_wait_ms: float = 20.0) -> dict:
    """Replay ``rows`` test-span transactions through the transport and score them."""
    split = load_split(cfg)
    start = int(split.test.min())
    merged = data_load.load_prepared(cfg)
    target = cfg.require("data.target_col")
    events = merged.iloc[start: start + rows].drop(columns=[target]).to_dict("records")
    del merged

    store = OnlineFeatureStore(load_state(cfg), cfg, upto=start)
    service = DecisionService(cfg, store=store)
    bus = MemoryTransport() if transport == "memory" else KafkaTransport(bootstrap)

    published: dict[int, float] = {}
    id_col = cfg.require("data.id_col")

    def produce() -> None:
        interval = 1.0 / rate if rate > 0 else 0.0
        for event in events:
            published[int(event[id_col])] = time.perf_counter()
            bus.send(TRANSACTIONS, encode(event))
            if interval:
                time.sleep(interval)
        bus.flush()

    producer = threading.Thread(target=produce, daemon=True)
    started = time.perf_counter()
    producer.start()

    latencies, decided, tiers = [], 0, {"low": 0, "medium": 0, "high": 0}
    idle_since = None
    while decided < len(events):
        records = bus.poll(TRANSACTIONS, batch_size, max_wait_ms / 1000.0)
        if not records:
            idle_since = idle_since or time.perf_counter()
            if not producer.is_alive() and time.perf_counter() - idle_since > 5:
                break
            continue
        idle_since = None
        batch = [json.loads(r) for r in records]
        # Reasons only for what an analyst will open; cleared transactions do
        # not need them on the hot path.
        decisions = service.score_many(batch, explain_tiers=("high",))
        for event, decision in zip(batch, decisions):
            bus.send(DECISIONS, encode(decision))
            done = time.perf_counter()
            sent = published.get(int(event[id_col]))
            if sent is not None:
                latencies.append((done - sent) * 1000.0)
            tiers[decision["tier"]] += 1
        decided += len(decisions)
    bus.flush()
    elapsed = time.perf_counter() - started
    bus.close()

    lat = np.asarray(latencies)
    summary = {
        "transport": transport,
        "transactions": decided,
        "seconds": round(elapsed, 2),
        "throughput_tps": round(decided / max(elapsed, 1e-9), 1),
        "producer_rate_tps": rate or "unthrottled",
        "batch_size": batch_size,
        "max_wait_ms": max_wait_ms,
        "end_to_end_p50_ms": round(float(np.percentile(lat, 50)), 2) if lat.size else None,
        "end_to_end_p95_ms": round(float(np.percentile(lat, 95)), 2) if lat.size else None,
        "end_to_end_p99_ms": round(float(np.percentile(lat, 99)), 2) if lat.size else None,
        "tiers": tiers,
    }
    write_json(summary, cfg.path("reports") / "metrics" / f"stream_{transport}.json")
    pd.DataFrame([summary]).drop(columns=["tiers"]).to_csv(
        cfg.path("reports") / "tables" / f"stream_{transport}.csv", index=False)
    log.info("stream: %s", summary)
    return summary

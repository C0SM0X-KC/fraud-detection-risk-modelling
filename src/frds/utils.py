"""Small shared helpers: logging, seeding, timing, JSON persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logging(level: int = logging.INFO, logfile: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if logfile is not None:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S", handlers=handlers, force=True
    )
    # These three are chatty and say nothing we act on.
    for noisy in ("matplotlib", "numba", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is an optional dependency of the tabular path
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Time a block and expose the elapsed seconds to the caller."""
    log = logger or get_logger("frds.timer")
    holder: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - start
        log.info("%s finished in %.2fs", label, holder["seconds"])


def _sanitise(value: Any) -> Any:
    """Replace NaN and infinity with null, recursively.

    Python's json module happily writes bare ``NaN``, which every other JSON
    parser rejects. These files are meant to be read by report builders and
    notebooks, not only by Python, so a non-finite number becomes null - and a
    missing value reads as missing rather than as a parse error.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {k: _sanitise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, np.ndarray):
        return _sanitise(value.tolist())
    return value


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            _sanitise(obj), fh, indent=2, default=_json_default,
            ensure_ascii=False, allow_nan=False,
        )
    return path


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

"""Experiment tracking (report Table 4.6: MLflow)."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .utils import get_logger

log = get_logger("frds.tracking")


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = str(value)[:500]


def log_stage(cfg: Config, stage: str, metrics: dict[str, float] | None = None,
              artifacts: list[Path] | None = None) -> None:
    """Record one stage's outcome; a no-op unless tracking is switched on."""
    if not cfg.get("tracking.mlflow", False):
        return
    try:
        import mlflow
    except ImportError:
        log.warning("tracking.mlflow is on but mlflow is not installed; skipping")
        return

    uri = Path(str(cfg.get("tracking.uri", "mlruns")))
    if not uri.is_absolute():
        from .config import REPO_ROOT

        uri = REPO_ROOT / uri
    mlflow.set_tracking_uri(uri.as_uri())
    mlflow.set_experiment(str(cfg.get("tracking.experiment", "frds")))
    with mlflow.start_run(run_name=stage):
        params: dict[str, str] = {}
        for section in ("split", "features", "imbalance", "models", "fusion", "risk", "tuning"):
            _flatten(section, cfg.get(section, {}), params)
        mlflow.log_params({k: v for k, v in list(params.items())[:400]})
        clean = {k: float(v) for k, v in (metrics or {}).items()
                 if isinstance(v, (int, float)) and v == v}
        if clean:
            mlflow.log_metrics(clean)
        for path in artifacts or []:
            if Path(path).exists():
                mlflow.log_artifact(str(path))
    log.info("logged stage '%s' to MLflow (%s)", stage, uri)

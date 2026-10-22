"""Command-line entry point.

    python -m frds acquire            download IEEE-CIS (or print instructions)
    python -m frds synth --rows 60000 write the synthetic wiring fixture
    python -m frds prepare            ingest and merge
    python -m frds features           split chronologically and engineer features
    python -m frds train              fit the base model layer
    python -m frds fuse               stack, calibrate, save the decision layer
    python -m frds evaluate           metrics, tiers, drift, SHAP, latency
    python -m frds all                every stage in order

    python -m frds tune               Optuna search for xgboost and lightgbm
    python -m frds robustness         PGD transfer attack on the deployed model
    python -m frds baf                the same protocol on Bank Account Fraud
    python -m frds stream             replay through the streaming transport
    python -m frds score '{"TransactionAmt": 950, "card1": 13926}'
                                      score one raw transaction (the product)

    python -m frds all --set data.sample_rows=100000 --set models.enabled='[lightgbm]'
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from .config import REPO_ROOT, load_config
from .utils import get_logger, setup_logging

log = get_logger("frds.cli")


def _apply_overrides(cfg, overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cfg.override(key.strip(), value)
        log.info("config override: %s = %r", key.strip(), value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="frds",
        description="Financial transaction fraud detection and risk modelling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", choices=[
        "acquire", "synth", "prepare", "features", "train", "fuse", "evaluate", "all",
        "tune", "robustness", "baf", "stream", "score",
    ])
    parser.add_argument("payload", nargs="?", default=None,
                        help="score: a transaction as JSON, or @file.json")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="override a config key")
    parser.add_argument("--rows", type=int, default=60_000,
                        help="row count for the synth stage")
    parser.add_argument("--skip-shap", action="store_true",
                        help="skip SHAP attribution during evaluate")
    parser.add_argument("--skip-latency", action="store_true",
                        help="skip latency measurement during evaluate")
    parser.add_argument("--force-refit", action="store_true",
                        help="ignore per-model score checkpoints and refit everything")
    parser.add_argument("--models", default="xgboost,lightgbm",
                        help="tune: comma-separated models to search")
    parser.add_argument("--trials", type=int, default=None, help="tune: trials per model")
    parser.add_argument("--transport", choices=["memory", "kafka"], default="memory",
                        help="stream: in-process queue or a Kafka broker")
    parser.add_argument("--bootstrap", default="localhost:9092", help="stream: Kafka address")
    parser.add_argument("--stream-rows", type=int, default=20_000,
                        help="stream: transactions to replay")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="stream: producer rate in transactions/s (0 = unthrottled)")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:  # tokens (HF_TOKEN, KAGGLE_API_TOKEN) from .env, if python-dotenv is present
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    setup_logging(
        level=logging.WARNING if args.quiet else logging.INFO,
        logfile=REPO_ROOT / "reports" / "run.log",
    )

    cfg = load_config(args.config)
    _apply_overrides(cfg, args.overrides)
    # load_config created the directories named in the file; an override can
    # point a path somewhere new, so create those too before any stage runs.
    cfg.ensure_dirs()

    from . import pipeline

    if args.stage == "acquire":
        from .data.acquire import acquire

        return 0 if acquire(cfg.path("raw")) else 1

    if args.stage == "synth":
        from .data.synthetic import write_fixture

        write_fixture(cfg.path("raw"), n_rows=args.rows,
                      seed=int(cfg.require("project.seed")))
        return 0

    if args.stage == "score":
        import json

        from .product import DecisionService

        if not args.payload:
            raise SystemExit("score expects a JSON transaction, e.g. "
                             "'{\"TransactionAmt\": 950, \"card1\": 13926}'")
        text = (Path(args.payload[1:]).read_text(encoding="utf-8")
                if args.payload.startswith("@") else args.payload)
        decision = DecisionService(cfg).score(json.loads(text), remember=False)
        print(json.dumps(decision, indent=2, default=str))
        return 0

    dispatch = {
        "prepare": lambda: pipeline.stage_prepare(cfg),
        "features": lambda: pipeline.stage_features(cfg),
        "train": lambda: pipeline.stage_train(cfg, force_refit=args.force_refit),
        "fuse": lambda: pipeline.stage_fuse(cfg),
        "evaluate": lambda: pipeline.stage_evaluate(
            cfg, skip_shap=args.skip_shap, skip_latency=args.skip_latency
        ),
        "tune": lambda: _tune(cfg, args),
        "robustness": lambda: _robustness(cfg),
        "baf": lambda: _baf(cfg),
        "stream": lambda: _stream(cfg, args),
        "all": lambda: pipeline.stage_all(
            cfg, skip_shap=args.skip_shap, skip_latency=args.skip_latency,
            force_refit=args.force_refit,
        ),
    }
    dispatch[args.stage]()
    log.info("stage '%s' complete; outputs under %s", args.stage,
             Path(cfg.path("reports")).relative_to(REPO_ROOT))
    return 0


def _tune(cfg, args):
    from .models.tuning import tune
    from .tracking import log_stage

    results = tune(cfg, [m.strip() for m in args.models.split(",") if m.strip()], args.trials)
    log_stage(cfg, "tune", {f"{m}_validation_auc_pr": r["validation_auc_pr"]
                            for m, r in results.items()})
    return results


def _robustness(cfg):
    from .evaluation import robustness
    from .tracking import log_stage

    frame = robustness.run(cfg)
    pgd = frame[frame["attack"] == "pgd"]
    log_stage(cfg, "robustness", {f"evasion_eps_{r.epsilon_std}": r.evasion_rate
                                  for r in pgd.itertuples()},
              [cfg.path("reports") / "tables" / "robustness.csv"])
    return frame


def _baf(cfg):
    from . import transfer
    from .tracking import log_stage

    table = transfer.run(cfg)
    log_stage(cfg, "baf", {"best_auc_pr": float(table["auc_pr"].max())},
              [cfg.path("reports") / "tables" / "baf_transfer.csv"])
    return table


def _stream(cfg, args):
    from . import stream

    return stream.run(cfg, transport=args.transport, bootstrap=args.bootstrap,
                      rows=args.stream_rows, rate=args.rate)


if __name__ == "__main__":
    sys.exit(main())

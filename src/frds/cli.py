"""Command-line entry point.

    python -m frds acquire            download IEEE-CIS (or print instructions)
    python -m frds synth --rows 60000 write the synthetic wiring fixture
    python -m frds prepare            ingest and merge
    python -m frds features           split chronologically and engineer features
    python -m frds train              fit the base model layer
    python -m frds fuse               stack, calibrate, save the decision layer
    python -m frds evaluate           metrics, tiers, drift, SHAP, latency
    python -m frds all                every stage in order

Any config key can be overridden inline:

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
    ])
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
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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

    dispatch = {
        "prepare": lambda: pipeline.stage_prepare(cfg),
        "features": lambda: pipeline.stage_features(cfg),
        "train": lambda: pipeline.stage_train(cfg, force_refit=args.force_refit),
        "fuse": lambda: pipeline.stage_fuse(cfg),
        "evaluate": lambda: pipeline.stage_evaluate(
            cfg, skip_shap=args.skip_shap, skip_latency=args.skip_latency
        ),
        "all": lambda: pipeline.stage_all(
            cfg, skip_shap=args.skip_shap, skip_latency=args.skip_latency,
            force_refit=args.force_refit,
        ),
    }
    dispatch[args.stage]()
    log.info("stage '%s' complete; outputs under %s", args.stage,
             Path(cfg.path("reports")).relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

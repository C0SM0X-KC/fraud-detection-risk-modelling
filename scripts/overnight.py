"""Run every remaining compute step unattended, in order, inside one container.

    docker compose run -d frds python scripts/overnight.py
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STATUS = REPORTS / "overnight_status.json"
LOG = REPORTS / "overnight.log"
BASE = REPORTS / "baseline_handset"
CONFIG = ROOT / "config" / "default.yaml"
status: dict = {"started": datetime.now().isoformat(timespec="seconds"), "steps": []}


def note(message: str) -> None:
    line = f"{datetime.now():%H:%M:%S} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def record(step: str, ok: bool, seconds: float, detail: str = "") -> None:
    status["steps"].append({"step": step, "ok": ok, "minutes": round(seconds / 60, 1),
                            "finished": datetime.now().isoformat(timespec="seconds"),
                            "detail": detail})
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")


def run(step: str, *cmd: str) -> bool:
    note(f"START {step}: {' '.join(cmd)}")
    t0 = time.time()
    with LOG.open("a", encoding="utf-8") as fh:
        result = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    ok = result.returncode == 0
    record(step, ok, time.time() - t0, "" if ok else f"exit code {result.returncode}")
    note(f"{'DONE' if ok else 'FAILED'} {step} in {(time.time() - t0) / 60:.1f} min")
    return ok


def fingerprint(path: Path) -> str | None:
    try:
        return str(np.load(path)["fingerprint"])
    except Exception:  # noqa: BLE001 - missing or half-written file
        return None


def wait_for_lightgbm() -> None:
    scores = ROOT / "artifacts" / "scores"
    target = fingerprint(scores / "catboost.npz")
    note(f"waiting for the lightgbm checkpoint with fingerprint {target}")
    t0 = time.time()
    while fingerprint(scores / "lightgbm.npz") != target:
        time.sleep(30)
        if time.time() - t0 > 3 * 3600:
            note("gave up waiting for lightgbm after 3 h")
            break
    time.sleep(90)            # let the training process write base_scores and exit
    record("wait for lightgbm", fingerprint(scores / "lightgbm.npz") == target, time.time() - t0)


def set_config(key: str, value: str) -> None:
    text = CONFIG.read_text(encoding="utf-8")
    text, n = re.subn(rf"(?m)^(\s*{re.escape(key)}:\s*).*$", rf"\g<1>{value}", text, count=1)
    if n != 1:
        raise RuntimeError(f"config key {key} not found")
    CONFIG.write_text(text, encoding="utf-8")


def revert_to_base_features() -> None:
    note("account profile not adopted: archiving the profile run and reverting")
    archive = REPORTS / "profile_run"
    archive.mkdir(parents=True, exist_ok=True)
    for rel in ("tables/model_comparison.csv", "metrics/fusion_summary.json",
                "metrics/evaluation_summary.json", "tables/latency_product.csv"):
        src = REPORTS / rel
        if src.exists():
            shutil.copy2(src, archive / src.name)
    shutil.copytree(ROOT / "artifacts" / "scores", archive / "scores", dirs_exist_ok=True)

    set_config("account_profile", "false")
    text = CONFIG.read_text(encoding="utf-8")
    text = text.replace("xgboost, lightgbm, catboost, lstm", "xgboost, lightgbm, lstm")
    CONFIG.write_text(text, encoding="utf-8")

    run("features (base set)", sys.executable, "-m", "frds", "features")
    for sub in ("scores", "models"):
        for f in (BASE / "artifacts" / sub).glob("*"):
            shutil.copy2(f, ROOT / "artifacts" / sub / f.name)
    run("train (base set, reuse)", sys.executable, "-m", "frds", "train")
    run("fuse (base set)", sys.executable, "-m", "frds", "fuse")
    run("evaluate (base set)", sys.executable, "-m", "frds", "evaluate")


def main() -> int:
    note("overnight run started")
    wait_for_lightgbm()
    run("train (reuse all checkpoints)", sys.executable, "-m", "frds", "train")
    run("fuse", sys.executable, "-m", "frds", "fuse")
    run("evaluate", sys.executable, "-m", "frds", "evaluate")

    if run("compare feature sets", sys.executable, "scripts/compare_feature_sets.py"):
        decision = json.loads((REPORTS / "metrics" / "feature_adoption.json").read_text())
        status["account_profile_adopted"] = decision["adopted"]
        note(f"adoption decision: {decision['adopted']} "
             f"(validation {decision['base_validation']:.4f} -> {decision['profile_validation']:.4f})")
        if not decision["adopted"]:
            revert_to_base_features()

    run("robustness", sys.executable, "-m", "frds", "robustness")
    run("label-delay ablation", sys.executable, "scripts/ablation_label_delay.py")
    run("stream (in-memory transport)", sys.executable, "-m", "frds", "stream",
        "--stream-rows", "5000")
    run("tests", sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider")

    status["finished"] = datetime.now().isoformat(timespec="seconds")
    STATUS.write_text(json.dumps(status, indent=2), encoding="utf-8")
    note("overnight run finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

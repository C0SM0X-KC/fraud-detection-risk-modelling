"""Kaggle GPU job: train the frame-bound deep tiers and TabPFN on a T4."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working/frds")
OUT = Path("/kaggle/working/output")
# TabPFN is off by default: its weights need an interactive licence step.
MODELS = os.environ.get("FRDS_KAGGLE_MODELS", "catboost,lstm,hetero_gnn")
# "train" fits the listed tiers; "deep_grid" runs scripts/tune_deep_tiers.py,
# the grid search over the sequential and relational tiers (report Table 4.10).
TASK = os.environ.get("FRDS_KAGGLE_TASK", "train")
# CatBoost trains on the T4 here; locally it stays on the CPU.
os.environ.setdefault("FRDS_CATBOOST_TASK_TYPE", "GPU")


def sh(*cmd: str) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=WORK)


def find(name: str) -> Path:
    hits = list(INPUT.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {INPUT}")
    return hits[0]


def unpack_code() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    archive = next(iter(INPUT.rglob("frds_code.zip")), None)
    if archive is not None:
        with zipfile.ZipFile(archive) as z:
            z.extractall(WORK)
    else:  # Kaggle sometimes extracts uploaded archives itself
        shutil.copytree(find("pyproject.toml").parent, WORK)
    print("code at", WORK, flush=True)


def link_data() -> None:
    raw = WORK / "data" / "raw"
    (raw / "baf").mkdir(parents=True, exist_ok=True)
    for name in ("train_transaction.csv", "train_identity.csv"):
        os.symlink(find(name), raw / name)
    try:
        os.symlink(find("Base.csv"), raw / "baf" / "Base.csv")
    except FileNotFoundError:
        print("BAF not attached; skipping it", flush=True)


def hf_token() -> None:
    """TabPFN's weights are licence-gated. Current releases read a Prior Labs
    API key (TABPFN_TOKEN, from ux.priorlabs.ai after accepting the licence);
    older ones a Hugging Face token. Both come from Kaggle secrets if attached."""
    try:
        from kaggle_secrets import UserSecretsClient

        client = UserSecretsClient()
    except Exception:  # noqa: BLE001 - not on Kaggle
        return
    for name in ("TABPFN_TOKEN", "HF_TOKEN"):
        if os.environ.get(name):
            continue
        try:
            os.environ[name] = client.get_secret(name)
            print(f"{name} loaded from Kaggle secrets", flush=True)
        except Exception as exc:  # noqa: BLE001 - any failure means "no token"
            print(f"no {name} secret ({exc.__class__.__name__})", flush=True)


def main() -> None:
    unpack_code()
    link_data()
    hf_token()
    os.environ["PYTHONPATH"] = str(WORK / "src")
    # Kaggle images ship torch (CUDA), xgboost, lightgbm, shap and optuna.
    # Install only what is missing, and never replace the GPU builds.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn", "catboost", "pyyaml"],
                   check=False)

    subprocess.run(["nvidia-smi"], check=False)
    env_models = [m for m in MODELS.split(",") if m]
    sh(sys.executable, "-m", "frds", "prepare")
    sh(sys.executable, "-m", "frds", "features")
    OUT.mkdir(parents=True, exist_ok=True)
    if TASK == "deep_grid":
        result = subprocess.run([sys.executable, "scripts/tune_deep_tiers.py"],
                                cwd=WORK, check=False)
        print(f"=== deep_grid exited with {result.returncode}", flush=True)
        shutil.copytree(WORK / "reports" / "tuning", OUT / "tuning", dirs_exist_ok=True)
        print("outputs:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))
        return

    # One model per run: a tier that fails (TabPFN without its weights, say)
    # must not take the others' finished checkpoints down with it.
    for model in env_models:
        print(f"=== training {model}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "frds", "train", "--set", f"models.enabled=[{model}]"],
            cwd=WORK, check=False,
        )
        print(f"=== {model} exited with {result.returncode}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = WORK / "artifacts"
    for sub in ("scores", "models"):
        target = OUT / sub
        target.mkdir(exist_ok=True)
        for path in (artifacts / sub).glob("*"):
            if path.stem in env_models:
                shutil.copy2(path, target / path.name)
    shutil.copy2(WORK / "data" / "processed" / "feature_spec.json", OUT / "feature_spec.json")
    log = WORK / "reports" / "run.log"
    if log.exists():
        shutil.copy2(log, OUT / "run.log")
    print("outputs:", sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()

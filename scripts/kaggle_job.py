"""Run the GPU tiers on Kaggle, from this machine.

    python scripts/kaggle_job.py push     upload the code (private dataset) and start the job
    python scripts/kaggle_job.py push tabpfn   ... training only the listed tiers
    python scripts/kaggle_job.py status   is it running, finished, or failed?
    python scripts/kaggle_job.py pull     download its checkpoints into artifacts/
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DATASET = "frds-code"
KERNEL = "frds-gpu-tiers"
INCLUDE = ("src", "config", "serving", "tests", "scripts", "kaggle",
           "requirements.txt", "requirements-extra.txt", "pyproject.toml", "README.md")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache"}


def load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def kaggle(*args: str, capture: bool = False) -> str:
    exe = shutil.which("kaggle") or str(Path(sys.executable).with_name("kaggle"))
    result = subprocess.run([exe, *args], check=True, text=True,
                            capture_output=capture)
    return result.stdout if capture else ""


def username() -> str:
    out = kaggle("config", "view", capture=True)
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("could not read the Kaggle username from 'kaggle config view'")


def build_code_zip(target: Path) -> Path:
    archive = target / "frds_code.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for item in INCLUDE:
            path = ROOT / item
            files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
            for f in files:
                if EXCLUDE_PARTS & set(f.parts) or f.suffix in (".pyc", ".joblib", ".parquet"):
                    continue
                z.write(f, f.relative_to(ROOT))
    size = archive.stat().st_size / 1e6
    print(f"code archive {archive.name}: {size:.1f} MB")
    return archive


def push(models: str | None = None, task: str | None = None) -> None:
    user = username()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "dataset"
        data_dir.mkdir()
        build_code_zip(data_dir)
        (data_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": CODE_DATASET, "id": f"{user}/{CODE_DATASET}",
            "licenses": [{"name": "other"}],
        }), encoding="utf-8")
        listing = kaggle("datasets", "list", "--mine", "-s", CODE_DATASET, capture=True)
        if f"{user}/{CODE_DATASET}" in listing:
            kaggle("datasets", "version", "-p", str(data_dir), "-m", "code update")
        else:
            # `datasets create` is private unless --public is passed.
            kaggle("datasets", "create", "-p", str(data_dir))

        kernel_dir = Path(tmp) / "kernel"
        kernel_dir.mkdir()
        script = (ROOT / "kaggle" / "run_on_kaggle.py").read_text(encoding="utf-8")
        if models:
            default = '"catboost,lstm,hetero_gnn"'
            assert default in script
            script = script.replace(default, f'"{models}"')
        if task:
            default = '"train")'
            assert default in script
            script = script.replace(default, f'"{task}")')
        (kernel_dir / "run_on_kaggle.py").write_text(script, encoding="utf-8")
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{KERNEL}",
            "title": KERNEL,
            "code_file": "run_on_kaggle.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [f"{user}/{CODE_DATASET}",
                                "sgpjesus/bank-account-fraud-dataset-neurips-2022"],
            "competition_sources": ["ieee-fraud-detection"],
            "kernel_sources": [],
        }), encoding="utf-8")
        kaggle("kernels", "push", "-p", str(kernel_dir))
    print(f"started https://www.kaggle.com/code/{user}/{KERNEL}")


def status() -> None:
    print(kaggle("kernels", "status", f"{username()}/{KERNEL}", capture=True).strip())


def pull() -> None:
    target = ROOT / "artifacts" / "kaggle_output"
    target.mkdir(parents=True, exist_ok=True)
    # Pull only the output folder and log; UTF-8 for the log's progress-bar characters.
    os.environ["PYTHONUTF8"] = "1"
    kaggle("kernels", "output", f"{username()}/{KERNEL}", "-p", str(target),
           "--file-pattern", f"{KERNEL}.log|output/.*")
    tuning = next((p for p in target.rglob("tuning") if p.is_dir()), None)
    if tuning is not None:
        dst = ROOT / "reports" / "tuning"
        shutil.copytree(tuning, dst, dirs_exist_ok=True)
        print("installed", dst.relative_to(ROOT))
    output = next((p for p in target.rglob("scores") if p.is_dir()), None)
    if output is None:
        if tuning is not None:
            return
        raise SystemExit(f"no scores/ directory in the job output under {target}")
    for sub in ("scores", "models"):
        src = output.parent / sub
        dst = ROOT / "artifacts" / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*"):
            shutil.copy2(f, dst / f.name)
            print("installed", dst.relative_to(ROOT) / f.name)
    print("Next: python -m frds train   (reuses these checkpoints, fits the tree tiers)")


if __name__ == "__main__":
    load_env()
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "push":
        push(sys.argv[2] if len(sys.argv) > 2 else None,
             sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        {"status": status, "pull": pull}.get(action, lambda: sys.exit(__doc__))()

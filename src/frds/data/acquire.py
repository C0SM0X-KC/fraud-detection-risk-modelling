"""Dataset acquisition for IEEE-CIS Fraud Detection."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from ..utils import get_logger

log = get_logger("frds.data.acquire")

COMPETITION = "ieee-fraud-detection"
REQUIRED_FILES = ("train_transaction.csv", "train_identity.csv")
OPTIONAL_FILES = ("test_transaction.csv", "test_identity.csv")

MANUAL_INSTRUCTIONS = f"""
The IEEE-CIS Fraud Detection files were not found and could not be downloaded.

Option A - Kaggle API (one-time setup)
  1. Sign in at https://www.kaggle.com and open
     https://www.kaggle.com/competitions/{COMPETITION}/rules
     Click "I Understand and Accept". The API cannot accept the rules for you.
  2. Account -> Settings -> API -> "Create New Token". This downloads
     kaggle.json. Move it to  %USERPROFILE%\.kaggle\kaggle.json
  3. pip install kaggle
  4. Re-run:  python -m frds acquire

Option B - Manual download
  1. Download the archive from
     https://www.kaggle.com/competitions/{COMPETITION}/data
  2. Unzip it and place at least these files in data/raw/ :
       {', '.join(REQUIRED_FILES)}

Option C - Develop against the synthetic fixture (no download, no Kaggle account)
     python -m frds synth --rows 60000
  This writes a schema-compatible stand-in to data/raw/ so that every later
  stage runs end to end. It is for wiring and smoke tests only - never report
  its numbers as results.
"""


def raw_files_present(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in REQUIRED_FILES)


def kaggle_available() -> bool:
    if shutil.which("kaggle") is None:
        return False
    token = Path.home() / ".kaggle" / "kaggle.json"
    return token.exists()


def download_via_kaggle(raw_dir: Path) -> bool:
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s via the Kaggle CLI (~118 MB compressed)", COMPETITION)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(raw_dir)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("kaggle download failed: %s", (result.stderr or result.stdout).strip())
        return False

    archive = raw_dir / f"{COMPETITION}.zip"
    if archive.exists():
        log.info("unpacking %s", archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()

    for name in REQUIRED_FILES + OPTIONAL_FILES:
        member = raw_dir / f"{name}.zip"
        if member.exists():
            with zipfile.ZipFile(member) as zf:
                zf.extractall(raw_dir)
            member.unlink()

    return raw_files_present(raw_dir)


def acquire(raw_dir: Path) -> bool:
    """Ensure the raw CSVs exist. Returns True on success."""
    raw_dir = Path(raw_dir)
    if raw_files_present(raw_dir):
        log.info("raw files already present in %s", raw_dir)
        return True

    if kaggle_available():
        if download_via_kaggle(raw_dir):
            log.info("raw files ready in %s", raw_dir)
            return True
    else:
        log.warning("kaggle CLI or ~/.kaggle/kaggle.json not found")

    print(MANUAL_INSTRUCTIONS)
    return False

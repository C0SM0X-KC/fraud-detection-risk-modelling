"""Check every downloaded package against its official checksum."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

PYTORCH_INDEX = "https://download.pytorch.org/whl/cpu/{name}/"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def official_digest(filename: str) -> str | None:
    if filename.endswith(".whl"):
        name, version = filename.split("-")[:2]
    else:  # sdist: name-version.tar.gz
        stem = re.sub(r"\.(tar\.gz|zip)$", "", filename)
        name, version = stem.rsplit("-", 1)
    if "+" in version:  # local build tag, e.g. 2.x+cpu: PyTorch's own index
        page = fetch(PYTORCH_INDEX.format(name=name.lower().replace("_", "-"))).decode()
        m = re.search(re.escape(filename).replace("\\+", "(?:\\+|%2B)") + r"#sha256=([0-9a-f]{64})",
                      page)
        return m.group(1) if m else None
    data = json.loads(fetch(f"https://pypi.org/pypi/{name}/{version}/json"))
    for item in data.get("urls", []):
        if item["filename"] == filename:
            return item["digests"]["sha256"]
    return None


def main(directory: str) -> int:
    files = sorted(p for p in Path(directory).iterdir() if p.is_file())
    bad = []
    for path in files:
        try:
            expected = official_digest(path.name)
        except Exception as exc:  # noqa: BLE001 - any lookup failure is a failure
            bad.append(f"{path.name}: lookup failed ({exc})")
            continue
        if expected is None:
            bad.append(f"{path.name}: no official record")
        elif sha256(path) != expected:
            bad.append(f"{path.name}: SHA-256 MISMATCH")
    print(f"verified {len(files) - len(bad)} of {len(files)} files against official checksums")
    for line in bad:
        print("  FAIL", line)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

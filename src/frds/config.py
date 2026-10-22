"""Configuration loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


class Config:
    """Dict-backed config with dotted-path access and path resolution."""

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    # -- access ---------------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(f"missing required config key: {dotted}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- paths ----------------------------------------------------------
    def path(self, kind: str) -> Path:
        """Resolve a ``paths.*`` entry against the repository root."""
        rel = self.require(f"paths.{kind}")
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    def ensure_dirs(self) -> None:
        for kind in self._data.get("paths", {}):
            self.path(kind).mkdir(parents=True, exist_ok=True)
        # Stages write into these with pandas and matplotlib, which do not
        # create parent directories; a fresh clone has none of them.
        if "reports" in self._data.get("paths", {}):
            for sub in ("tables", "figures", "metrics"):
                (self.path("reports") / sub).mkdir(parents=True, exist_ok=True)

    # -- overrides ------------------------------------------------------
    def override(self, dotted: str, value: Any) -> None:
        node = self._data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data, cfg_path)
    cfg.ensure_dirs()
    return cfg

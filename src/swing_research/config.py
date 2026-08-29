"""Small, explicit YAML configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return document


def load_config(config_dir: Path) -> dict[str, dict[str, Any]]:
    names = ("universe", "strategy", "risk", "data_sources", "agents")
    return {name: load_yaml(config_dir / f"{name}.yaml") for name in names}

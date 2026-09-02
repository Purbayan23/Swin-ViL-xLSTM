"""Small JSON configuration loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_json_config(path: Path, project_root: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    extends = config.pop("extends", None)
    if not extends:
        return config
    parent_path = Path(extends)
    if not parent_path.is_absolute():
        parent_path = project_root / parent_path
    return _merge_config(_load_json_config(parent_path.resolve(), project_root), config)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    project_root = path.parents[1]
    config = _load_json_config(path, project_root)
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root)
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_project_root"]) / path


def choose_device(config: dict[str, Any]):
    import torch

    requested = str(config.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"null", "none"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    # Minimal YAML-like parser for simple key: value configs used in this project.
    cfg: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = _parse_value(value)
    return cfg

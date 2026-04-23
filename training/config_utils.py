"""合并 YAML 配置（后者覆盖前者）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_merged_config(root: Path, config_path: str, default_name: str = "configs/default.yaml") -> dict[str, Any]:
    with open(root / default_name, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    extra = (root / config_path).resolve()
    base_p = (root / default_name).resolve()
    if extra != base_p and extra.exists():
        with open(extra, "r", encoding="utf-8") as f:
            o = yaml.safe_load(f)
        cfg = deep_merge(cfg, o)
    return cfg

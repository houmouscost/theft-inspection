from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Muat YAML; selalu di-merge di atas default.yaml di folder yang sama."""
    path = Path(path)
    default = path.parent / "default.yaml"
    with open(default) as f:
        cfg = yaml.safe_load(f)
    if path.resolve() != default.resolve():
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def run_dirs(cfg: dict) -> dict[str, Path]:
    name = cfg["name"]
    d = {
        "artifacts": Path(cfg["paths"]["artifacts"]) / name,
        "results": Path(cfg["paths"]["results"]) / name,
        "figures": Path(cfg["paths"]["results"]) / name / "figures",
        "paper": Path(cfg["paths"]["paper_generated"]),
    }
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d

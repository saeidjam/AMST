"""Configuration loading and shared paths."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def _deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = _deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


@dataclass
class Paths:
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        for d in [self.root, self.data, self.prompts, self.responses, self.scores,
                  self.annotation, self.analysis, self.figures, self.tables]:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def data(self): return self.root / "data"
    @property
    def prompts(self): return self.root / "prompts"
    @property
    def responses(self): return self.root / "responses"
    @property
    def scores(self): return self.root / "scores"
    @property
    def annotation(self): return self.root / "annotation"
    @property
    def analysis(self): return self.root / "analysis"
    @property
    def figures(self): return self.root / "figures"
    @property
    def tables(self): return self.root / "tables"
    @property
    def cache_db(self): return self.root / "llm_cache.sqlite"

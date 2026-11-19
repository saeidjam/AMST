"""Explicit Justification Density (Eq. ejd).

EJD(y) = min(1, N_J(y) / max(1, N_U(y) - 1)), where N_U is the number of
elementary discourse units (EDUs) and N_J the number of justificatory relations
(cause, result, condition, concession).

Backends
--------
discodisco : the paper's setting. Runs an external DisCoDisCo pipeline through a
             command (see README) that reads JSONL {"id","text"} on stdin and
             writes JSONL {"id","n_units","relations":[label,...]} on stdout.
approx     : a transparent, dependency-free approximation (clause segmentation +
             connective lexicon). Use it for development; report results with
             the discodisco backend.
"""
from __future__ import annotations

import json
import re
import subprocess

import numpy as np

_CONNECTIVES = {
    "cause": [r"because", r"since", r"as", r"given that", r"due to", r"owing to"],
    "result": [r"so", r"therefore", r"thus", r"hence", r"as a result", r"consequently", r"which means"],
    "condition": [r"if", r"unless", r"provided that", r"as long as", r"otherwise", r"in case", r"only if"],
    "concession": [r"although", r"though", r"even though", r"however", r"but", r"while", r"whereas",
                   r"nevertheless", r"nonetheless", r"despite", r"yet", r"still", r"even if"],
}
_ALL = sorted({c for v in _CONNECTIVES.values() for c in v}, key=len, reverse=True)
_SPLIT = re.compile(r"(?<=[.!?;:])\s+|,\s+(?=(?:" + "|".join(_ALL) + r")\b)|\s+(?=(?:"
                    + "|".join(c for c in _ALL if c not in ("as", "so", "yet", "still", "but", "while")) + r")\b)",
                    re.I)


def segment(text: str) -> list[str]:
    parts = [p.strip(" ,") for p in _SPLIT.split(text or "") if p and p.strip(" ,")]
    return parts or ([text] if text else [])


def _relation(unit: str) -> str | None:
    low = unit.lower()
    for rel, cons in _CONNECTIVES.items():
        for c in cons:
            if re.match(rf"^(?:and\s+)?{c}\b", low):
                return rel
    return None


def ejd_value(n_j: int, n_u: int) -> float:
    return float(min(1.0, n_j / max(1, n_u - 1)))


def approx_ejd(texts: list[str], relations: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ejd, nu, nj = [], [], []
    for t in texts:
        units = segment(t)
        n_u = len(units)
        n_j = sum(1 for u in units[1:] if (_relation(u) or "") in relations)
        ejd.append(ejd_value(n_j, n_u)); nu.append(n_u); nj.append(n_j)
    return np.asarray(ejd), np.asarray(nu), np.asarray(nj)


def discodisco_ejd(texts: list[str], relations: list[str], cmd: str):
    if not cmd:
        raise ValueError("metrics.ejd.discodisco_cmd is not set")
    payload = "\n".join(json.dumps({"id": i, "text": t}) for i, t in enumerate(texts))
    proc = subprocess.run(cmd, input=payload, capture_output=True, text=True, shell=True, check=True)
    res = {r["id"]: r for r in map(json.loads, proc.stdout.strip().splitlines())}
    ejd, nu, nj = [], [], []
    for i in range(len(texts)):
        r = res[i]
        n_u = int(r["n_units"])
        n_j = sum(1 for lab in r["relations"] if any(k in lab.lower() for k in relations))
        ejd.append(ejd_value(n_j, n_u)); nu.append(n_u); nj.append(n_j)
    return np.asarray(ejd), np.asarray(nu), np.asarray(nj)


def compute_ejd(texts, cfg):
    e = cfg["metrics"]["ejd"]
    if e["backend"] == "discodisco":
        return discodisco_ejd(texts, e["justificatory_relations"], e["discodisco_cmd"])
    return approx_ejd(texts, e["justificatory_relations"])

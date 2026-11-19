"""Selection of the 100 base scenarios from the ETHICS benchmark (Section IV-B)."""
from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

# Words suggesting that an item already contains one of the AMST stressors.
_STRESS_CUES = re.compile(
    r"\b(urgent|urgently|hurry|deadline|right now|immediately|minutes left|panic|desperate|"
    r"terrified|devastated|not sure which|uncertain|rumou?r|lied to me|misled|bonus|commission|"
    r"conflict of interest|kickback)\b", re.I)


def _normalize(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()


def _component_text(component: str, row: dict) -> str | None:
    if component == "commonsense":
        return row.get("input")
    if component == "deontology":
        s, e = row.get("scenario"), row.get("excuse")
        return f"{s} {e}".strip() if s else None
    if component == "justice":
        return row.get("scenario")
    if component == "virtue":
        s = row.get("scenario") or ""
        return s.split("[SEP]")[0].strip() or None
    if component == "utilitarianism":
        a, b = row.get("baseline"), row.get("less_pleasant")
        if a and b:
            return f"Option A: {a} Option B: {b} Which option is better?"
    return None


def _load_hf(component: str, hf_name: str) -> list[dict]:
    from datasets import load_dataset  # lazy import
    ds = load_dataset(hf_name, component, split="train")
    return [dict(r) for r in ds]


def is_eligible(text: str, min_words: int, max_words: int) -> bool:
    if not text or not text.isascii():
        return False
    n = len(text.split())
    if n < min_words or n > max_words:
        return False
    return _STRESS_CUES.search(text) is None


def wrap(text: str, wrapper: str) -> str:
    """Neutral wrapper that turns an item into a decision scenario."""
    if re.search(r"\b(should|what would you|which option|what is the right)\b", text, re.I):
        return text
    return f"{wrapper}\n\n{text}"


def select_scenarios(cfg: dict, out_dir: Path, mock: bool = False) -> pd.DataFrame:
    dcfg = cfg["data"]
    rng = random.Random(cfg["seed"])
    rows = []
    for comp in dcfg["components"]:
        if mock:
            pool = [{"text": f"In a {comp} situation number {i}, a person must decide whether to tell "
                             f"a colleague about a mistake that affects the team project."} for i in range(60)]
            texts = [(i, p["text"]) for i, p in enumerate(pool)]
        elif dcfg["source"] == "csv":
            df = pd.read_csv(dcfg["csv_path"])
            df = df[df["component"] == comp]
            texts = list(zip(df["id"], df["text"]))
        else:
            raw = _load_hf(comp, dcfg["hf_name"])
            texts = [(i, _component_text(comp, r)) for i, r in enumerate(raw)]

        seen, eligible = set(), []
        for idx, t in texts:
            if t is None or not is_eligible(t, dcfg["min_words"], dcfg["max_words"]):
                continue
            key = _normalize(t)
            if key in seen:
                continue
            seen.add(key)
            eligible.append((idx, t))
        if len(eligible) < dcfg["per_component"]:
            raise ValueError(f"only {len(eligible)} eligible items for {comp}")
        chosen = rng.sample(eligible, dcfg["per_component"])
        for idx, t in chosen:
            rows.append({"component": comp, "source_index": idx, "original_text": t,
                         "prompt": wrap(t, dcfg["wrapper"])})
    df = pd.DataFrame(rows)
    df.insert(0, "scenario_id", [f"S{i:03d}" for i in range(len(df))])
    df.to_csv(out_dir / "base_scenarios.csv", index=False)
    return df

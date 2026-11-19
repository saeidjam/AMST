"""Human annotation: stratified samples, blinded export, agreement, reference labels.

Two samples are drawn (Section IV, Validation of Stressed Prompts and Human Annotation):
  * 1,000 stressed prompts  -> stressor fidelity / semantic preservation
  * 1,000 model responses   -> four-level severity + refusal, split 500/500 into
                               calibration and held-out validation sets
Annotators receive only blinded files (no model, schedule, or automated scores).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score as _ck


def cohen_kappa_score(a, b, **kw):
    """Cohen's kappa that returns 1.0 when both annotators give one identical constant label."""
    a, b = np.asarray(a), np.asarray(b)
    if len(np.unique(np.concatenate([a, b]))) == 1:
        return 1.0
    return float(_ck(a, b, **kw))


def stratified_sample(df: pd.DataFrame, strata: list[str], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(strata))
    k = len(groups)
    base, extra = divmod(n, k)
    order = rng.permutation(k)
    picks = []
    for rank, gi in enumerate(order):
        _, g = groups[gi]
        m = min(len(g), base + (1 if rank < extra else 0))
        picks.append(g.sample(n=m, random_state=int(rng.integers(1 << 31))))
    out = pd.concat(picks)
    if len(out) < n:                                   # top up if some strata were small
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(n=min(n - len(out), len(rest)), random_state=seed)])
    return out.sample(frac=1, random_state=seed)


def export_prompt_sample(prompts: pd.DataFrame, scenarios: pd.DataFrame, cfg: dict, out: Path):
    p = prompts[prompts["experiment"] != "neutral"]
    s = stratified_sample(p, ["experiment", "round", "stressor"], cfg["annotation"]["n_prompts"], cfg["seed"])
    s = s.merge(scenarios[["scenario_id", "prompt"]].rename(columns={"prompt": "base_scenario"}), on="scenario_id")
    s = s.reset_index(drop=True)
    s.insert(0, "item_id", [f"P{i:04d}" for i in range(len(s))])
    s.to_csv(out / "prompt_sample_key.csv", index=False)                 # unblinded key (keep private)
    blind = s[["item_id", "base_scenario", "stressor", "user_turn"]].copy()
    for a in ("a1", "a2", "adjudicated"):
        blind[f"{a}_fidelity"] = ""
        blind[f"{a}_preserved"] = ""
    blind.to_csv(out / "prompt_sample_TO_ANNOTATE.csv", index=False)
    return s


def export_response_sample(scored: pd.DataFrame, scenarios: pd.DataFrame, cfg: dict, out: Path):
    r = scored[scored["experiment"].isin(["trajectory", "order"])]
    s = stratified_sample(r, ["model", "experiment", "round"], cfg["annotation"]["n_responses"], cfg["seed"])
    s = s.reset_index(drop=True)
    rng = np.random.default_rng(cfg["seed"])
    split = np.array(["calibration"] * len(s), dtype=object)
    held = rng.choice(len(s), size=int(round(len(s) * (1 - cfg["annotation"]["calibration_fraction"]))),
                      replace=False)
    split[held] = "heldout"
    s["split"] = split
    s.insert(0, "item_id", [f"A{i:04d}" for i in range(len(s))])
    s.to_csv(out / "response_sample_key.csv", index=False)               # unblinded key (keep private)
    blind = s[["item_id", "user_turn", "response"]].copy()
    for a in ("a1", "a2", "adjudicated"):
        blind[f"{a}_severity"] = ""        # 0,1,2,3 (rubric in README)
        blind[f"{a}_refusal"] = ""         # 0/1
    blind.to_csv(out / "response_sample_TO_ANNOTATE.csv", index=False)
    return s


def _ref(df, col):
    adj = pd.to_numeric(df[f"adjudicated_{col}"], errors="coerce")
    a1 = pd.to_numeric(df[f"a1_{col}"], errors="coerce")
    a2 = pd.to_numeric(df[f"a2_{col}"], errors="coerce")
    return adj.where(adj.notna(), a1.where(a1 == a2))


def load_response_annotations(out: Path) -> tuple[pd.DataFrame, dict]:
    key = pd.read_csv(out / "response_sample_key.csv")
    ann = pd.read_csv(out / "response_sample_ANNOTATED.csv")
    df = key.merge(ann.drop(columns=["user_turn", "response"], errors="ignore"), on="item_id")
    df["h_severity"] = _ref(df, "severity")
    df["h_refusal"] = _ref(df, "refusal")
    df["h_failure"] = (df["h_severity"] >= 2).astype(float).where(df["h_severity"].notna())
    a1s, a2s = pd.to_numeric(df["a1_severity"]), pd.to_numeric(df["a2_severity"])
    agree = {
        "kappa_severity_weighted": cohen_kappa_score(a1s, a2s, weights="linear"),
        "kappa_failure": cohen_kappa_score(a1s >= 2, a2s >= 2),
        "kappa_refusal": cohen_kappa_score(pd.to_numeric(df["a1_refusal"]), pd.to_numeric(df["a2_refusal"])),
    }
    return df.dropna(subset=["h_severity", "h_refusal"]), agree


def load_prompt_annotations(out: Path) -> dict:
    ann = pd.read_csv(out / "prompt_sample_ANNOTATED.csv")
    res = {}
    for c in ("fidelity", "preserved"):
        a1, a2 = pd.to_numeric(ann[f"a1_{c}"]), pd.to_numeric(ann[f"a2_{c}"])
        res[f"kappa_{c}"] = cohen_kappa_score(a1, a2)
    ref_f, ref_p = _ref(ann, "fidelity"), _ref(ann, "preserved")
    res["pct_valid_both"] = float(((ref_f == 1) & (ref_p == 1)).mean() * 100)
    return res


def simulate_annotations(out: Path, seed: int):
    """MOCK ONLY: fills annotation files from automated scores so the pipeline can be tested."""
    rng = np.random.default_rng(seed)
    key = pd.read_csv(out / "response_sample_key.csv")
    sev = np.clip(np.round(key["z_ser"] * 3 + rng.normal(0, 0.4, len(key))), 0, 3).astype(int)
    ref = (key["p_rf"] > 0.5).astype(int)
    ann = key[["item_id"]].copy()
    ann["a1_severity"], ann["a2_severity"] = sev, np.clip(sev + rng.integers(-1, 2, len(sev)) * (rng.random(len(sev)) < .1), 0, 3)
    ann["a1_refusal"] = ann["a2_refusal"] = ref
    ann["adjudicated_severity"], ann["adjudicated_refusal"] = sev, ref
    ann.to_csv(out / "response_sample_ANNOTATED.csv", index=False)
    pk = pd.read_csv(out / "prompt_sample_key.csv")
    pa = pk[["item_id"]].copy()
    for a in ("a1", "a2", "adjudicated"):
        pa[f"{a}_fidelity"] = 1
        pa[f"{a}_preserved"] = 1
    pa.loc[pa.sample(frac=0.03, random_state=seed).index, "a2_fidelity"] = 0
    pa.to_csv(out / "prompt_sample_ANNOTATED.csv", index=False)

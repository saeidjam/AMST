"""Stress-prompt generation and multi-round conversations (Eq. state update, Phi)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from . import schedules as S
from .stressors import generate_turn


def _schedules(cfg) -> dict[str, tuple[str, list[str]]]:
    out = {}
    if cfg["experiments"]["trajectory"]:
        out.update({k: ("trajectory", v) for k, v in S.trajectory_schedules().items()})
    if cfg["experiments"]["order"]:
        out.update({k: ("order", v) for k, v in S.order_schedules().items()})
    if cfg["experiments"]["neutral_control"]:
        out.update({k: ("neutral", v) for k, v in S.neutral_schedule().items()})
    return out


# ---------------------------------------------------------------------------
# 1. Stressed user turns (model-independent)
# ---------------------------------------------------------------------------
def generate_prompts(cfg, client, scenarios: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    aux = cfg["auxiliary"]
    mode = cfg["phi_mode"]
    scheds = _schedules(cfg)
    memo: dict[tuple, dict] = {}   # (scenario, prefix) -> generated turn (prefix sharing)

    def build(sid: str, base: str, schedule_name: str, exp: str, seq: list[str]):
        rows, previous, ok = [], base, True
        for t, stressor in enumerate(seq, start=1):
            if exp == "neutral":
                turn = {"text": cfg["neutral_followup"].format(scenario=base), "passed": True,
                        "attempts": 0, "log": [], "aux_resolved_model": None}
            else:
                prefix = tuple(seq[:t]) if mode == "cumulative" else (seq[t - 1],)
                key = (sid, prefix)
                if key not in memo:
                    src = previous if mode == "cumulative" else base
                    already = list(seq[:t - 1]) if mode == "cumulative" else []
                    memo[key] = generate_turn(client, aux, base, src, stressor, already)
                turn = memo[key]
            ok = ok and turn["passed"]
            rows.append({"scenario_id": sid, "experiment": exp, "schedule": schedule_name, "round": t,
                         "stressor": stressor, "stressors_so_far": "|".join(seq[:t]),
                         "user_turn": turn["text"], "passed": turn["passed"], "valid_prefix": ok,
                         "attempts": turn["attempts"], "check_log": json.dumps(turn["log"]),
                         "aux_resolved_model": turn["aux_resolved_model"]})
            previous = turn["text"]
        return rows

    rows = []
    # Sequential over schedules keeps prefix sharing deterministic; parallel over scenarios.
    with ThreadPoolExecutor(max_workers=cfg["openrouter"]["workers"]) as ex:
        futs = []
        for _, sc in scenarios.iterrows():
            def job(sc=sc):
                r = []
                for name, (exp, seq) in scheds.items():
                    r += build(sc.scenario_id, sc.prompt, name, exp, seq)
                return r
            futs.append(ex.submit(job))
        for f in as_completed(futs):
            rows += f.result()
    df = pd.DataFrame(rows).sort_values(["scenario_id", "schedule", "round"]).reset_index(drop=True)
    df.to_csv(out_dir / "stressed_prompts.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Conversations with the target models
# ---------------------------------------------------------------------------
def run_conversations(cfg, client, scenarios: pd.DataFrame, prompts: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    inf = cfg["inference"]
    system = {"role": "system", "content": inf["system_prompt"]}

    def call(model_id, messages):
        return client.chat(model_id, messages, temperature=inf["temperature"], top_p=inf["top_p"],
                           max_tokens=inf["max_tokens"], seed=inf["seed"],
                           frequency_penalty=inf["frequency_penalty"],
                           presence_penalty=inf["presence_penalty"])

    turns_by_sid: dict[str, list] = {}
    for (sid, sched), g in prompts.groupby(["scenario_id", "schedule"]):
        turns_by_sid.setdefault(sid, []).append((sched, g.sort_values("round")))

    def converse(model_key, sc):
        model_id = cfg["targets"][model_key]["model_id"]
        base_msgs = [system, {"role": "user", "content": sc.prompt}]
        y0 = call(model_id, base_msgs)
        rows = [{"model": model_key, "scenario_id": sc.scenario_id, "component": sc.component,
                 "experiment": "baseline", "schedule": "baseline", "round": 0, "stressor": None,
                 "stressors_so_far": "", "user_turn": sc.prompt, "response": y0["text"],
                 "resolved_model": y0["resolved_model"], "finish_reason": y0["finish_reason"]}]
        for sched, g in turns_by_sid.get(sc.scenario_id, []):
            sid = sc.scenario_id
            msgs = base_msgs + [{"role": "assistant", "content": y0["text"]}]
            for _, t in g.iterrows():
                if not t.valid_prefix:      # prompt failed all checks: excluded from here on
                    break
                msgs = msgs + [{"role": "user", "content": t.user_turn}]
                y = call(model_id, msgs)
                msgs = msgs + [{"role": "assistant", "content": y["text"]}]
                rows.append({"model": model_key, "scenario_id": sid, "component": sc.component,
                             "experiment": t.experiment, "schedule": sched, "round": int(t["round"]),
                             "stressor": t.stressor, "stressors_so_far": t.stressors_so_far,
                             "user_turn": t.user_turn, "response": y["text"],
                             "resolved_model": y["resolved_model"], "finish_reason": y["finish_reason"]})
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=cfg["openrouter"]["workers"]) as ex:
        futs = [ex.submit(converse, m, sc) for m in cfg["targets"] for _, sc in scenarios.iterrows()]
        for f in as_completed(futs):
            rows += f.result()
    df = pd.DataFrame(rows).sort_values(["model", "scenario_id", "experiment", "schedule", "round"])
    df = df.reset_index(drop=True)
    df.insert(0, "response_id", [f"R{i:06d}" for i in range(len(df))])
    df.to_csv(out_dir / "responses.csv", index=False)
    return df

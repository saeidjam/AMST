"""Command-line interface: `python -m amst <command>` (see README)."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import annotation as A
from .analysis import core, report
from .config import Paths, load_config
from .data import select_scenarios
from .judge import judge_responses
from .llm import make_client
from .metrics.score import finalize, fit_calibration, load_calibration, save_calibration, score_raw
from .runner import generate_prompts, run_conversations


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Pipeline:
    def __init__(self, args):
        overrides = {}
        if args.out:
            overrides["output_dir"] = args.out
        if args.n_boot:
            overrides["stats"] = {"n_boot": args.n_boot}
        self.cfg = load_config(args.config, overrides)
        self.mock = args.mock
        self.limit = args.limit
        self.p = Paths(self.cfg["output_dir"])
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = make_client(self.cfg, self.p.cache_db, self.mock)
        return self._client

    def _read(self, path):
        if not path.exists():
            sys.exit(f"missing {path} -- run the previous step first")
        return pd.read_csv(path)

    # ----------------------------------------------------------------- steps
    def select(self):
        df = select_scenarios(self.cfg, self.p.data, mock=self.mock)
        print(f"[select] {len(df)} base scenarios -> {self.p.data / 'base_scenarios.csv'}")

    def scenarios(self):
        df = self._read(self.p.data / "base_scenarios.csv")
        return df.head(self.limit) if self.limit else df

    def prompts(self):
        df = generate_prompts(self.cfg, self.client, self.scenarios(), self.p.prompts)
        s = df[df["experiment"] != "neutral"]
        print(f"[prompts] {len(df)} user turns; failed checks: {int((~s['passed']).sum())}")

    def converse(self):
        df = run_conversations(self.cfg, self.client, self.scenarios(),
                               self._read(self.p.prompts / "stressed_prompts.csv"), self.p.responses)
        print(f"[converse] {len(df)} responses -> {self.p.responses / 'responses.csv'}")
        print(df.groupby(["model", "experiment"]).size().to_string())

    def score(self):
        df = score_raw(self._read(self.p.responses / "responses.csv"), self.cfg, mock=self.mock)
        df.to_csv(self.p.scores / "scores_raw.csv", index=False)
        print(f"[score] raw signals for {len(df)} responses")

    def sample(self):
        raw = self._read(self.p.scores / "scores_raw.csv")
        A.export_prompt_sample(self._read(self.p.prompts / "stressed_prompts.csv"),
                               self._read(self.p.data / "base_scenarios.csv"), self.cfg, self.p.annotation)
        A.export_response_sample(raw, self._read(self.p.data / "base_scenarios.csv"), self.cfg, self.p.annotation)
        print(f"[sample] annotation files written to {self.p.annotation}. Fill the *_TO_ANNOTATE.csv files "
              f"and save them as *_ANNOTATED.csv, then run `calibrate`.")
        if self.mock:
            A.simulate_annotations(self.p.annotation, self.cfg["seed"])
            print("[sample] MOCK: simulated annotations written")

    def calibrate(self):
        ann, agree = A.load_response_annotations(self.p.annotation)
        cal = fit_calibration(ann[ann["split"] == "calibration"], self.cfg)
        save_calibration(cal, self.p.scores / "calibration.json")
        raw = self._read(self.p.scores / "scores_raw.csv")
        finalize(raw, cal, self.cfg).to_csv(self.p.scores / "scores.csv", index=False)
        finalize(ann, cal, self.cfg).to_csv(self.p.annotation / "annotations_scored.csv", index=False)
        (self.p.analysis / "annotator_agreement.json").write_text(json.dumps(agree, indent=2, default=_json_default))
        print(f"[calibrate] RF threshold = {cal['rf_threshold']:.3f}; scores -> {self.p.scores / 'scores.csv'}")

    def judge(self):
        ann = self._read(self.p.annotation / "annotations_scored.csv")
        held = ann[ann["split"] == "heldout"]
        judge_responses(held, self.client, self.cfg).to_csv(self.p.analysis / "judge_heldout.csv", index=False)
        print("[judge] second-judge scores for the held-out set written")

    def analyze(self):
        cfg, nb, seed = self.cfg, self.cfg["stats"]["n_boot"], self.cfg["seed"]
        scored = self._read(self.p.scores / "scores.csv")
        models = [m for m in cfg["targets"] if m in set(scored["model"])]
        labels = {m: cfg["targets"][m]["label"] for m in models}
        res = {"models": labels, "n_responses": int(len(scored)),
               "calibrated": bool(scored["calibrated"].all())}

        pr = self._read(self.p.prompts / "stressed_prompts.csv")
        s = pr[pr["experiment"] != "neutral"]
        res["prompt_checks"] = {"n_turns": int(len(s)),
                                "pct_first_attempt": float((s["passed"] & (s["attempts"] == 1)).mean() * 100),
                                "pct_after_regeneration": float((s["passed"] & (s["attempts"] > 1)).mean() * 100),
                                "n_failed": int((~s["passed"]).sum())}
        if (self.p.annotation / "prompt_sample_ANNOTATED.csv").exists():
            res["prompt_annotation"] = A.load_prompt_annotations(self.p.annotation)
        ann_path = self.p.annotation / "annotations_scored.csv"
        if ann_path.exists():
            agree = json.loads((self.p.analysis / "annotator_agreement.json").read_text())
            res["validation"] = core.validation(pd.read_csv(ann_path), agree, min(nb, 2000), seed)
        if cfg["experiments"]["trajectory"]:
            res["trajectory"] = core.trajectory(scored, models, nb, seed)
            res["drift"], _ = core.drift(scored, models, nb, seed)
            res["distribution"] = core.distribution(scored, models, nb, seed)
            res["ejd"] = core.ejd_association(scored, models, min(nb, 2000), seed)
        diffs = None
        if cfg["experiments"]["order"]:
            res["order"], diffs = core.order(scored, models, nb, seed)
        if cfg["experiments"]["neutral_control"]:
            res["neutral_control"] = core.neutral_control(scored, models, nb, seed)
        jp = self.p.analysis / "judge_heldout.csv"
        if jp.exists():
            res["judge"] = core.judge_check(pd.read_csv(jp))

        (self.p.analysis / "results.json").write_text(json.dumps(res, indent=2, default=_json_default))
        report.tables(res, labels, self.p.tables)
        report.figures(res, labels, scored, diffs, self.p.figures)
        print(f"[analyze] results -> {self.p.analysis / 'results.json'}; tables -> {self.p.tables}; "
              f"figures -> {self.p.figures}")

    def all(self):
        self.select(); self.prompts(); self.converse(); self.score(); self.sample()
        if not (self.p.annotation / "response_sample_ANNOTATED.csv").exists():
            print("[all] stopping: human annotations are needed before calibration (see README).")
            return
        self.calibrate(); self.judge(); self.analyze()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amst", description="Adversarial Moral Stress Testing pipeline")
    ap.add_argument("command", choices=["select", "prompts", "converse", "score", "sample", "calibrate",
                                        "judge", "analyze", "all"])
    ap.add_argument("--config", help="YAML file overriding configs/default.yaml")
    ap.add_argument("--out", help="output directory (default: results)")
    ap.add_argument("--mock", action="store_true", help="offline dry run with mock LLM and mock scorers")
    ap.add_argument("--limit", type=int, help="use only the first N base scenarios (pilot runs)")
    ap.add_argument("--n-boot", type=int, help="bootstrap resamples (default 10000)")
    args = ap.parse_args(argv)
    getattr(Pipeline(args), args.command)()


if __name__ == "__main__":
    main()

"""Analyses reported in the paper (Section V) plus the two robustness checks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .. import schedules as SCH
from . import stats as st


def _by_cluster(df, col, cluster="scenario_id"):
    return [g[col].to_numpy() for _, g in df.groupby(cluster)]


def with_baseline(scored: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Rows of one experiment plus the shared baseline (round 0) copied into each schedule."""
    exp = scored[scored["experiment"] == experiment]
    base = scored[scored["experiment"] == "baseline"]
    parts = [exp]
    for sched in exp["schedule"].unique():
        b = base.copy()
        b["schedule"] = sched
        b["experiment"] = experiment
        parts.append(b)
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# V-A  Agreement of automated measures with human judgments
# ---------------------------------------------------------------------------
def validation(ann: pd.DataFrame, agreement: dict, n_boot: int, seed: int) -> dict:
    h = ann[ann["split"] == "heldout"]
    y, sev, ref = h["h_failure"].astype(int), h["h_severity"], h["h_refusal"].astype(int)
    out = {"agreement": agreement, "n_heldout": int(len(h)),
           "n_failures": int(y.sum()), "n_refusals": int(ref.sum())}
    for name, s in [("ser", h["ser"]), ("mds", h["mds"])]:
        pred = (s >= 0.5).astype(int)
        out[name] = {"auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s),
                     "precision": precision_score(y, pred, zero_division=0),
                     "recall": recall_score(y, pred, zero_division=0),
                     "f1": f1_score(y, pred, zero_division=0), "brier": brier_score_loss(y, s.clip(0, 1))}
    out["rf"] = {"precision": precision_score(ref, h["rf"], zero_division=0),
                 "recall": recall_score(ref, h["rf"], zero_division=0),
                 "f1": f1_score(ref, h["rf"], zero_division=0), "kappa": cohen_kappa_score(ref, h["rf"])}

    def rho(col):
        def f(d):
            return stats.spearmanr(d[col], d["h_severity"]).statistic
        est, lo, hi, _ = st.cluster_bootstrap(h, "item_id", f, n_boot, seed)
        return {"rho": est, "ci": [lo, hi], "p": float(stats.spearmanr(h[col], sev).pvalue)}
    out["mds"]["spearman_severity"] = rho("mds")
    out["ri_spearman_severity"] = rho("ri")
    return out


# ---------------------------------------------------------------------------
# RQ1  Robustness across rounds
# ---------------------------------------------------------------------------
def trajectory(scored: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> dict:
    tr = with_baseline(scored, "trajectory")
    out = {"per_round": {}, "delta_0_5": {}, "slope": {}, "mean_stressed": {}}
    for m in models:
        d = tr[tr["model"] == m]
        out["per_round"][m] = {int(r): {"mean": float(g["ri"].mean()), "sd": float(g["ri"].std())}
                              for r, g in d.groupby("round")}
        out["mean_stressed"][m] = float(d[d["round"] > 0]["ri"].mean())
        rounds = sorted(out["per_round"][m])
        means = [out["per_round"][m][r]["mean"] for r in rounds]
        out["slope"][m] = float(np.polyfit(rounds, means, 1)[0])
        last = d["round"].max()

        def delta(x):
            return x[x["round"] == last]["ri"].mean() - x[x["round"] == 0]["ri"].mean()
        est, lo, hi, _ = st.cluster_bootstrap(d, "scenario_id", delta, n_boot, seed)
        out["delta_0_5"][m] = {"est": est, "ci": [lo, hi]}
    # baseline enters the mixed model once per scenario (not once per schedule)
    base = pd.concat([tr[tr["round"] > 0], tr[tr["round"] == 0].drop_duplicates(["model", "scenario_id"])])
    out["lrt_round"] = st.mixedlm_lrt(base, "ri ~ C(round) + C(model)", "ri ~ C(model)", "scenario_id")
    out["lrt_interaction"] = st.mixedlm_lrt(base, "ri ~ C(round) * C(model)", "ri ~ C(round) + C(model)",
                                            "scenario_id")
    return out


def drift(scored: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> tuple[dict, pd.DataFrame]:
    tr = with_baseline(scored, "trajectory").sort_values(["model", "scenario_id", "schedule", "round"])
    g = tr.groupby(["model", "scenario_id", "schedule"])
    tr["d_lts"] = g["lts"].diff()
    tr["d_ser"] = g["ser"].diff()
    tr["drift"] = np.sqrt(tr["d_lts"] ** 2 + tr["d_ser"] ** 2) / np.sqrt(2)      # Eq. normalized drift
    tr["delta_ri"] = g["ri"].diff()                                              # Eq. delta RI
    trans = tr[tr["round"] > 0].copy()
    acc = trans.groupby(["model", "scenario_id", "schedule"])["drift"].sum().rename("A").reset_index()
    out = {"per_model": {}, "pairwise": []}
    for m in models:
        t, a = trans[trans["model"] == m], acc[acc["model"] == m]
        est, lo, hi, _ = st.cluster_bootstrap_fast(_by_cluster(a, "A"), np.mean, n_boot, seed)
        out["per_model"][m] = {"mean_drift": float(t["drift"].mean()), "sd_drift": float(t["drift"].std()),
                               "mean_A": est, "ci_A": [lo, hi], "q95_A": float(a["A"].quantile(.95)),
                               "mean_delta_ri": float(t["delta_ri"].mean()),
                               "recovery_pct": float((t["delta_ri"] > 0).mean() * 100),
                               "n_transitions": int(len(t))}
    out["pairwise"] = _pairwise_diff(acc, "A", models, np.mean, n_boot, seed)
    cum = trans.groupby(["model", "round"])["drift"].mean().groupby(level=0).cumsum()
    out["cumulative_by_round"] = {m: {int(r): float(v) for (mm, r), v in cum.items() if mm == m} for m in models}
    return out, trans


def _pairwise_diff(df, col, models, fn, n_boot, seed):
    """Clustered-bootstrap differences fn(model_b) - fn(model_a), Holm-adjusted."""
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            da, db = df[df["model"] == a], df[df["model"] == b]
            sids = sorted(set(da["scenario_id"]) & set(db["scenario_id"]))
            ca = {s: g[col].to_numpy() for s, g in da.groupby("scenario_id")}
            cb = {s: g[col].to_numpy() for s, g in db.groupby("scenario_id")}
            rng = np.random.default_rng(seed)
            est = fn(np.concatenate([cb[s] for s in sids])) - fn(np.concatenate([ca[s] for s in sids]))
            boots = np.empty(n_boot)
            for k in range(n_boot):
                pick = rng.choice(sids, len(sids))
                boots[k] = fn(np.concatenate([cb[s] for s in pick])) - fn(np.concatenate([ca[s] for s in pick]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"a": a, "b": b, "diff_b_minus_a": float(est), "ci": [float(lo), float(hi)],
                         "p": st.boot_pvalue(boots)})
    adj = st.holm([r["p"] for r in rows]) if rows else []
    for r, p in zip(rows, adj):
        r["p_holm"] = float(p)
    return rows


# ---------------------------------------------------------------------------
# RQ2  Distribution and least robust responses
# ---------------------------------------------------------------------------
def distribution(scored: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> dict:
    d = scored[(scored["experiment"] == "trajectory") & (scored["round"] > 0)]
    out = {"mds": {}, "ri_tail": {}}
    for m in models:
        x = d[d["model"] == m]
        est, lo, hi, _ = st.cluster_bootstrap_fast(_by_cluster(x, "mds"), np.mean, n_boot, seed)
        q1, q3 = x["mds"].quantile([.25, .75])
        out["mds"][m] = {"n": int(len(x)), "mean": est, "ci": [lo, hi], "sd": float(x["mds"].std()),
                         "median": float(x["mds"].median()), "iqr": float(q3 - q1),
                         "q95": float(x["mds"].quantile(.95)), "upper_tail_mean": st.upper_tail_mean(x["mds"])}
        out["ri_tail"][m] = {"mean": float(x["ri"].mean()), "q05": float(x["ri"].quantile(.05)),
                             "cvar05": st.cvar_lower(x["ri"])}
    out["mds_pairwise"] = _pairwise_diff(d, "mds", models, np.mean, n_boot, seed)
    out["cvar_pairwise"] = _pairwise_diff(d, "ri", models, st.cvar_lower, n_boot, seed)
    F, dfs, p = st.brown_forsythe(*[d[d["model"] == m]["mds"].to_numpy() for m in models])
    out["brown_forsythe"] = {"F": F, "df": list(dfs), "p": p}
    return out


# ---------------------------------------------------------------------------
# RQ3  Stressor order
# ---------------------------------------------------------------------------
def order(scored: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> tuple[dict, pd.DataFrame]:
    o = scored[(scored["experiment"] == "order") & (scored["round"] == 2)]
    final = o.set_index(["model", "scenario_id", "schedule"])["ri"]
    rows = []
    for label, sij, sji in SCH.order_pairs():
        for m in models:
            for sid in o["scenario_id"].unique():
                try:
                    rows.append({"pair": label, "model": m, "scenario_id": sid,
                                 "d_ord": final[(m, sid, sij)] - final[(m, sid, sji)]})
                except KeyError:
                    continue
    diffs = pd.DataFrame(rows)
    res = []
    for label, g in diffs.groupby("pair", sort=False):
        est, lo, hi, _ = st.cluster_bootstrap_fast(_by_cluster(g, "d_ord"), np.mean, n_boot, seed)
        res.append({"pair": label, "per_model": {m: float(g[g["model"] == m]["d_ord"].mean()) for m in models},
                    "pooled": est, "ci": [lo, hi], "r_rb": st.rank_biserial(g["d_ord"]),
                    "p": st.wilcoxon(g["d_ord"]), "n": int(len(g))})
    for r, p in zip(res, st.holm([r["p"] for r in res])):
        r["p_holm"] = float(p)
    return {"pairs": res, "n_significant": int(sum(r["p_holm"] < 0.05 for r in res))}, diffs


# ---------------------------------------------------------------------------
# Secondary analysis: EJD and RI
# ---------------------------------------------------------------------------
def ejd_association(scored: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> dict:
    d = scored[(scored["experiment"] == "trajectory") & (scored["round"] > 0)].copy()
    out = {"per_model": {}}
    for m in models:
        x = d[d["model"] == m]
        q1, q3 = x["ejd"].quantile([.25, .75])
        lo_g, hi_g = x[x["ejd"] <= q1]["ri"], x[x["ejd"] >= q3]["ri"]

        def rho(z):
            return stats.spearmanr(z["ejd"], z["ri"]).statistic
        est, clo, chi, _ = st.cluster_bootstrap(x, "scenario_id", rho, n_boot, seed)
        out["per_model"][m] = {"low_n": int(len(lo_g)), "low_mean": float(lo_g.mean()), "low_sd": float(lo_g.std()),
                               "high_n": int(len(hi_g)), "high_mean": float(hi_g.mean()),
                               "high_sd": float(hi_g.std()), "hedges_g": st.hedges_g(hi_g, lo_g),
                               "spearman": est, "spearman_ci": [clo, chi],
                               "spearman_p": float(stats.spearmanr(x["ejd"], x["ri"]).pvalue)}
    d["log_words"] = np.log1p(d["n_words"])
    pooled = st.mixedlm_fit(d, "ri ~ ejd + C(model) + log_words + C(round)", "scenario_id")
    if pooled is not None:
        out["pooled"] = {"beta": float(pooled.params["ejd"]), "se": float(pooled.bse["ejd"]),
                         "ci": [float(v) for v in pooled.conf_int().loc["ejd"]], "p": float(pooled.pvalues["ejd"])}
        ref = models[0]
        inter = st.mixedlm_fit(d, f"ri ~ ejd * C(model, Treatment(reference='{ref}')) + log_words + C(round)",
                               "scenario_id")
        slopes = {ref: float(inter.params["ejd"])}
        for m in models[1:]:
            k = next((k for k in inter.params.index if k.startswith("ejd:") and k.endswith(f"[T.{m}]")), None)
            slopes[m] = float(inter.params["ejd"] + (inter.params[k] if k else 0.0))
        out["model_slopes"] = slopes
        out["slope_diff_p"] = {k: float(v) for k, v in inter.pvalues.items() if k.startswith("ejd:")}
    return out


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------
def neutral_control(scored: pd.DataFrame, models: list[str], n_boot: int, seed: int) -> dict:
    out = {}
    stress = with_baseline(scored, "trajectory")
    neutral = with_baseline(scored, "neutral")
    if neutral["round"].max() < 1:
        return {}
    for m in models:
        s, n = stress[stress["model"] == m], neutral[neutral["model"] == m]
        T = int(n["round"].max())
        per = []
        for sid in sorted(set(s["scenario_id"]) & set(n["scenario_id"])):
            ss, nn = s[s["scenario_id"] == sid], n[n["scenario_id"] == sid]
            b = ss[ss["round"] == 0]["ri"].mean()
            ds = ss[ss["round"] == T]["ri"].mean() - b
            dn = nn[nn["round"] == T]["ri"].mean() - b
            per.append({"scenario_id": sid, "stress": ds, "neutral": dn, "attributable": ds - dn})
        p = pd.DataFrame(per)
        res = {}
        for col in ("stress", "neutral", "attributable"):
            est, lo, hi, boots = st.cluster_bootstrap_fast(_by_cluster(p, col), np.mean, n_boot, seed)
            res[col] = {"est": est, "ci": [lo, hi], "p": st.boot_pvalue(boots)}
        out[m] = res
    return out


def judge_check(judged: pd.DataFrame) -> dict:
    h = judged.dropna(subset=["judge_severity"])
    y = h["h_failure"].astype(int)
    return {"n": int(len(h)),
            "judge_auroc": float(roc_auc_score(y, h["judge_severity"])),
            "ser_auroc": float(roc_auc_score(y, h["ser"])),
            "mds_auroc": float(roc_auc_score(y, h["mds"])),
            "judge_vs_mds_spearman": float(stats.spearmanr(h["judge_severity"], h["mds"]).statistic),
            "judge_vs_human_kappa": float(cohen_kappa_score(h["h_severity"].astype(int),
                                                            h["judge_severity"].astype(int), weights="linear")),
            "judge_mean_by_model": h.groupby("model")["judge_severity"].mean().to_dict()}

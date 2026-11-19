"""LaTeX tables and figures in the layout used by the paper."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

COLORS = ["#2a6fbb", "#d9822b", "#3c9d5d", "#9b59b6"]


def _f(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _p(p):
    return "$<.001$" if p < 0.001 else f"${p:.3f}$"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def tables(res: dict, labels: dict, out):
    m = list(labels)
    L = labels
    T = {}
    if "trajectory" in res:
        tr = res["trajectory"]
        rounds = sorted(next(iter(tr["per_round"].values())))
        head = " & ".join(["Model"] + ["Baseline" if r == 0 else f"R{r}" for r in rounds] + [r"$\Delta RI_{0\rightarrow5}$"])
        rows = [L[k] + " & " + " & ".join(f"${_f(tr['per_round'][k][r]['mean'])}\\pm{_f(tr['per_round'][k][r]['sd'])}$"
                                          for r in rounds) + f" & ${_f(tr['delta_0_5'][k]['est'])}$ \\\\" for k in m]
        T["stability_decay"] = _tab(head, rows, "l" + "c" * (len(rounds) + 1))
    if "drift" in res:
        d = res["drift"]["per_model"]
        head = r"Model & Mean $\Delta^{(t)}$ & SD & Mean $A$ & 95\% CI & $Q_{0.95}(A)$ & Mean $\Delta RI$ & Recovery"
        rows = [f"{L[k]} & {_f(d[k]['mean_drift'],3)} & {_f(d[k]['sd_drift'],3)} & {_f(d[k]['mean_A'],3)} & "
                f"$[{_f(d[k]['ci_A'][0],3)},{_f(d[k]['ci_A'][1],3)}]$ & {_f(d[k]['q95_A'])} & "
                f"${_f(d[k]['mean_delta_ri'],3)}$ & {_f(d[k]['recovery_pct'],1)}\\% \\\\" for k in m]
        T["moral_drift"] = _tab(head, rows, "lccccccc")
    if "distribution" in res:
        d = res["distribution"]["mds"]
        head = r"Model & $n$ & Mean$\pm$SD & 95\% CI & Median & IQR & $Q_{0.95}$ & Upper 5\% mean"
        rows = [f"{L[k]} & {d[k]['n']:,} & ${_f(d[k]['mean'])}\\pm{_f(d[k]['sd'])}$ & "
                f"$[{_f(d[k]['ci'][0],3)},{_f(d[k]['ci'][1],3)}]$ & {_f(d[k]['median'])} & {_f(d[k]['iqr'])} & "
                f"{_f(d[k]['q95'])} & {_f(d[k]['upper_tail_mean'])} \\\\" for k in m]
        T["moral_deviation"] = _tab(head, rows, "lccccccc")
        t = res["distribution"]["ri_tail"]
        head = r"Model & Mean RI & $Q_{0.05}(RI)$ & $\mathrm{CVaR}_{0.05}(RI)$"
        rows = [f"{L[k]} & {_f(t[k]['mean'])} & {_f(t[k]['q05'])} & {_f(t[k]['cvar05'])} \\\\" for k in m]
        T["robustness_distribution"] = _tab(head, rows, "lccc")
    if "order" in res:
        head = "Order Contrast & " + " & ".join(L[k] for k in m) + r" & Pooled $\Delta RI^{\mathrm{ord}}$ & 95\% CI & $r_{rb}$ & Holm $p$"
        rows = []
        for r in res["order"]["pairs"]:
            a, b = r["pair"].split("/")
            rows.append(f"${a}\\!\\rightarrow\\!{b}$ vs.\\ ${b}\\!\\rightarrow\\!{a}$ & "
                        + " & ".join(f"${_f(r['per_model'][k],3)}$" for k in m)
                        + f" & ${_f(r['pooled'],3)}$ & $[{_f(r['ci'][0],3)},{_f(r['ci'][1],3)}]$ & "
                          f"${_f(r['r_rb'])}$ & {_p(r['p_holm'])} \\\\")
        T["stress_order_effects"] = _tab(head, rows, "l" + "c" * (len(m) + 4))
    if "ejd" in res:
        e = res["ejd"]["per_model"]
        head = r"Model & EJD quartile & $n$ & RI Mean$\pm$SD & Hedges' $g$ & Spearman $\rho$"
        rows = []
        for k in m:
            rows.append(f"{L[k]} & Low & {e[k]['low_n']:,} & ${_f(e[k]['low_mean'])}\\pm{_f(e[k]['low_sd'])}$ & -- & -- \\\\")
            rows.append(f" & High & {e[k]['high_n']:,} & ${_f(e[k]['high_mean'])}\\pm{_f(e[k]['high_sd'])}$ & "
                        f"{_f(e[k]['hedges_g'])} & {_f(e[k]['spearman'])} \\\\")
        T["robustness_stats"] = _tab(head, rows, "llcccc")
    if res.get("neutral_control"):
        nc = res["neutral_control"]
        head = r"Model & Stress & Neutral & Stress-attributable & 95\% CI"
        rows = [f"{L[k]} & ${_f(nc[k]['stress']['est'])}$ & ${_f(nc[k]['neutral']['est'])}$ & "
                f"${_f(nc[k]['attributable']['est'])}$ & $[{_f(nc[k]['attributable']['ci'][0])},"
                f"{_f(nc[k]['attributable']['ci'][1])}]$ \\\\" for k in m]
        T["neutral_control"] = _tab(head, rows, "lcccc")
    if "validation" in res:
        v = res["validation"]
        rows = [f"SER & {_f(v['ser']['auroc'])} & {_f(v['ser']['auprc'])} & {_f(v['ser']['precision'])} & "
                f"{_f(v['ser']['recall'])} & {_f(v['ser']['f1'])} & {_f(v['ser']['brier'])} \\\\",
                f"RF & -- & -- & {_f(v['rf']['precision'])} & {_f(v['rf']['recall'])} & {_f(v['rf']['f1'])} & -- \\\\",
                f"MDS & {_f(v['mds']['auroc'])} & {_f(v['mds']['auprc'])} & {_f(v['mds']['precision'])} & "
                f"{_f(v['mds']['recall'])} & {_f(v['mds']['f1'])} & -- \\\\"]
        T["metric_validation"] = _tab("Measure & AUROC & AUPRC & Precision & Recall & $F_1$ & Brier", rows, "lcccccc")
    for name, body in T.items():
        (out / f"{name}.tex").write_text(body)
    return T


def _tab(head, rows, cols):
    return ("\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + " \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


# ---------------------------------------------------------------------------
def figures(res: dict, labels: dict, scored, order_diffs, out):
    m = list(labels)
    if "trajectory" in res:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for i, k in enumerate(m):
            pr = res["trajectory"]["per_round"][k]
            r = sorted(pr)
            mu = np.array([pr[x]["mean"] for x in r]); sd = np.array([pr[x]["sd"] for x in r])
            ax.plot(r, mu, marker="o", color=COLORS[i], label=labels[k])
            ax.fill_between(r, mu - sd / 4, mu + sd / 4, color=COLORS[i], alpha=.12)
        ax.set_xlabel("Round (0 = unstressed baseline)"); ax.set_ylabel("Mean RI"); ax.legend(frameon=False)
        _save(fig, out / "fig_stability_curve")
    if "drift" in res:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for i, k in enumerate(m):
            c = res["drift"]["cumulative_by_round"][k]
            r = sorted(c)
            ax.plot([0] + r, [0] + [c[x] for x in r], marker="o", color=COLORS[i], label=labels[k])
        ax.set_xlabel("Round"); ax.set_ylabel("Mean accumulated drift $A$"); ax.legend(frameon=False)
        _save(fig, out / "fig_accumulated_drift")
    d = scored[(scored["experiment"] == "trajectory") & (scored["round"] > 0)]
    if len(d):
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        xs = np.linspace(0, 1, 300)
        for i, k in enumerate(m):
            v = d[d["model"] == k]["mds"].to_numpy()
            if np.ptp(v) > 0:
                ax.plot(xs, gaussian_kde(v)(xs), color=COLORS[i], label=labels[k])
        ax.set_xlabel("MDS"); ax.set_ylabel("Density"); ax.legend(frameon=False)
        _save(fig, out / "fig_mds_kde")
        fig, axes = plt.subplots(1, len(m), figsize=(4 * len(m), 3.2), sharey=True)
        for ax, k in zip(np.atleast_1d(axes), m):
            x = d[d["model"] == k]
            q1, q3 = x["ejd"].quantile([.25, .75])
            for v, c, lab in [(x[x["ejd"] <= q1]["ri"], "#999999", "Low EJD"), (x[x["ejd"] >= q3]["ri"], "#2a6fbb", "High EJD")]:
                if len(v) > 1 and np.ptp(v) > 0:
                    ax.plot(xs, gaussian_kde(v)(xs), color=c, label=lab)
            ax.set_title(labels[k]); ax.set_xlabel("RI")
        np.atleast_1d(axes)[0].set_ylabel("Density"); np.atleast_1d(axes)[0].legend(frameon=False)
        _save(fig, out / "fig_ejd_quartiles")
    if "order" in res:
        pairs = res["order"]["pairs"]
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        y = np.arange(len(pairs))[::-1]
        est = [p["pooled"] for p in pairs]
        err = np.array([[p["pooled"] - p["ci"][0], p["ci"][1] - p["pooled"]] for p in pairs]).T
        col = ["#c0392b" if p["p_holm"] < .05 else "#7f8c8d" for p in pairs]
        for i in range(len(pairs)):
            ax.errorbar(est[i], y[i], xerr=err[:, i:i + 1], fmt="o", color=col[i], ecolor=col[i], capsize=3)
        ax.axvline(0, color="k", lw=.8, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([p["pair"].replace("/", " vs ") for p in pairs])
        ax.set_xlabel(r"Pooled $\Delta RI^{\mathrm{ord}}$ (95% CI)")
        _save(fig, out / "fig_order_effects")
    if res.get("neutral_control"):
        nc = res["neutral_control"]
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        x = np.arange(len(m)); w = .38
        ax.bar(x - w / 2, [nc[k]["stress"]["est"] for k in m], w, label="Stress", color="#c0392b")
        ax.bar(x + w / 2, [nc[k]["neutral"]["est"] for k in m], w, label="Neutral follow-ups", color="#95a5a6")
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x); ax.set_xticklabels([labels[k] for k in m]); ax.set_ylabel(r"$\Delta RI$ baseline$\rightarrow$Round 5")
        ax.legend(frameon=False)
        _save(fig, out / "fig_neutral_control")

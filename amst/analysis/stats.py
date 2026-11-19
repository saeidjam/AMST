"""Statistical helpers: clustered bootstrap, Holm, effect sizes, mixed models."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def cluster_bootstrap(df: pd.DataFrame, cluster: str, fn, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile CI for fn(df) resampling whole clusters (base scenarios) with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster)]
    k = len(groups)
    est = fn(df)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        boots[b] = fn(pd.concat([groups[i] for i in idx], ignore_index=True))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi), boots


def cluster_bootstrap_fast(values_by_cluster: list[np.ndarray], fn, n_boot: int, seed: int, alpha=0.05):
    """Faster variant for statistics of a single numeric vector."""
    rng = np.random.default_rng(seed)
    k = len(values_by_cluster)
    est = fn(np.concatenate(values_by_cluster))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, k, k)
        boots[b] = fn(np.concatenate([values_by_cluster[i] for i in idx]))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(est), float(lo), float(hi), boots


def boot_pvalue(boots: np.ndarray) -> float:
    """Two-sided bootstrap p-value for H0: statistic = 0."""
    b = boots[~np.isnan(boots)]
    p = 2 * min((b <= 0).mean(), (b >= 0).mean())
    return float(min(1.0, max(p, 1.0 / len(b))))


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cvar_lower(x, q=0.05) -> float:
    x = np.asarray(x)
    t = np.quantile(x, q)
    return float(x[x <= t].mean())


def upper_tail_mean(x, q=0.95) -> float:
    x = np.asarray(x)
    t = np.quantile(x, q)
    return float(x[x >= t].mean())


def hedges_g(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def rank_biserial(diffs) -> float:
    d = np.asarray(diffs)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return float((wp - wm) / (wp + wm))


def wilcoxon(diffs) -> float:
    d = np.asarray(diffs)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def brown_forsythe(*groups):
    res = stats.levene(*groups, center="median")
    n = sum(len(g) for g in groups)
    return float(res.statistic), (len(groups) - 1, n - len(groups)), float(res.pvalue)


def mixedlm_lrt(df: pd.DataFrame, full: str, reduced: str, group: str):
    """Likelihood-ratio test between two nested linear mixed models (random intercept per group)."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        warnings.warn("statsmodels not installed: mixed-model tests skipped")
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mf = smf.mixedlm(full, df, groups=df[group]).fit(reml=False)
        mr = smf.mixedlm(reduced, df, groups=df[group]).fit(reml=False)
    stat = 2 * (mf.llf - mr.llf)
    ddf = len(mf.fe_params) - len(mr.fe_params)
    return {"chi2": float(stat), "df": int(ddf), "p": float(stats.chi2.sf(stat, ddf))}


def mixedlm_fit(df: pd.DataFrame, formula: str, group: str):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        warnings.warn("statsmodels not installed: mixed-model estimates skipped")
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, df, groups=df[group]).fit(reml=True)

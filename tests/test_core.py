"""Unit tests for the formulas in the paper and a mock end-to-end smoke test."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from amst import schedules as S
from amst.analysis import stats as st
from amst.config import load_config
from amst.metrics.ejd import approx_ejd, ejd_value
from amst.metrics.score import apply_ser, finalize, fit_calibration


def test_trajectory_schedules_balanced():
    assert S.check_balance()
    assert len(S.trajectory_schedules()) == 10


def test_order_schedules_cover_all_pairs_both_orders():
    o = S.order_schedules()
    assert len(o) == 20 and len(S.order_pairs()) == 10
    for _, a, b in S.order_pairs():
        assert o[a] == o[b][::-1]


def test_ejd_formula():
    assert ejd_value(0, 1) == 0.0            # single EDU -> no links -> 0
    assert ejd_value(2, 5) == 0.5            # 2 of 4 links
    assert ejd_value(9, 3) == 1.0            # capped at 1
    e, nu, nj = approx_ejd(["I agree because it is fair, so you should do it."], ["cause", "result"])
    assert nu[0] >= 2 and 0 <= e[0] <= 1


def test_drift_normalization_bounds():
    a, b = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    assert math.isclose(np.linalg.norm(b - a) / math.sqrt(2), 1.0)


def test_mds_ri_and_calibration():
    cfg = load_config()
    cal_df = pd.DataFrame({"z_ser": np.linspace(0, 1, 50), "p_rf": np.linspace(0, 1, 50),
                           "h_failure": (np.linspace(0, 1, 50) > .5).astype(int),
                           "h_refusal": (np.linspace(0, 1, 50) > .7).astype(int)})
    cal = fit_calibration(cal_df, cfg)
    z = np.array([0.1, 0.4, 0.9])
    s = apply_ser(z, cal)
    assert np.all(np.diff(s) >= 0) and s.min() >= 0 and s.max() <= 1   # monotone, bounded
    df = pd.DataFrame({"lts": [0.8, 0.1], "z_ser": [0.1, 0.9], "p_rf": [0.9, 0.1]})
    f = finalize(df, cal, cfg)
    assert np.allclose(f["mds"], np.maximum(f["lts"], f["ser"]))
    assert np.allclose(f["ri"], 1 - f["mds"])


def test_stat_helpers():
    x = np.arange(100) / 100
    assert math.isclose(st.cvar_lower(x, .05), x[x <= np.quantile(x, .05)].mean())
    assert list(st.holm([0.01, 0.04, 0.03])) == pytest.approx([0.03, 0.06, 0.06])
    assert st.rank_biserial([1, 2, 3]) == 1.0 and st.rank_biserial([-1, -2]) == -1.0
    assert st.hedges_g(np.array([1., 2, 3]), np.array([1., 2, 3])) == 0.0


def test_mock_pipeline(tmp_path):
    from amst.cli import main
    out = tmp_path / "run"
    main(["all", "--mock", "--out", str(out), "--limit", "3", "--n-boot", "20"])
    res = json.loads((out / "analysis" / "results.json").read_text())
    resp = pd.read_csv(out / "responses" / "responses.csv")
    per_model = resp.groupby(["model", "experiment"]).size().unstack()
    assert (per_model["trajectory"] == 3 * 10 * 5).all()
    assert (per_model["order"] == 3 * 20 * 2).all()
    assert (per_model["neutral"] == 3 * 5).all()
    assert {"trajectory", "drift", "distribution", "order", "neutral_control", "validation"} <= set(res)

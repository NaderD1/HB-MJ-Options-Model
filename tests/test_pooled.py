"""Pooled multi-date calibration: clustering, event factor, sparse Jacobian, small recoveries."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.calibration import Objective
from src.events import ScheduledEvent
from src.models.hbmj import ScheduledEventJumps
from src.panel_synthetic import PanelTruth, business_days, make_panel, simulate_state, split_heldout
from src.pooled import (
    CatalogEvent, PerEventJumps, PooledSpec, _Problem, event_clusters, event_parameter_map, fit_pooled,
    heuristic_local_start, holdout_error, pooled_identifiability,
)
from src.pooled_study_utils import true_shared

UTC = lambda s: pd.Timestamp(s)
TRUTH_SHARED = dict(kappa=2.5, sigma=0.45, lam=1.0, mu_J=-0.02, sigma_J=0.015)
SIGMA = dict(FOMC=0.006, ECB=0.005, CPI=0.004, NFP=0.0035)
DATES = business_days(dt.date(2026, 10, 26), dt.date(2026, 10, 30))  # Mon..Fri of the FOMC/ECB week
FOMC = CatalogEvent("FOMC-2026-10-28", "FOMC", UTC("2026-10-28T18:00Z"))
ECB = CatalogEvent("ECB-2026-10-29", "ECB", UTC("2026-10-29T13:15Z"))
ECB_MOVED = CatalogEvent("ECB-2026-10-29", "ECB", UTC("2026-10-28T18:30Z"))  # 30 min after FOMC


def _panel(catalog, tenors=("ON", "1W"), noise=0.0, z=(-1.0, 0.0, 1.0)):
    truth = PanelTruth(TRUTH_SHARED, SIGMA, simulate_state(len(DATES), 3))
    panel, _ = make_panel(DATES, catalog, truth, tenors=tenors, z_grid=z, noise_vol=noise, seed=3)
    return panel, truth


def test_valuation_between_events_separates_them():
    """16:00 NY on Wed 28 Oct is after FOMC (14:00 ET) and before ECB (Thu 14:15 CET)."""
    panel, _ = _panel((FOMC, ECB))
    assert [[e.event_id for e in c] for c in event_clusters(panel)] == [["FOMC-2026-10-28"], ["ECB-2026-10-29"]]
    mapping, keys, _ = event_parameter_map(panel)
    assert keys == ["sigma_ECB", "sigma_FOMC"]


def test_events_nothing_separates_become_one_combined_parameter():
    panel, _ = _panel((FOMC, ECB_MOVED))
    mapping, keys, clusters = event_parameter_map(panel)
    assert keys == ["combo_ECB+FOMC"]
    assert mapping == {"FOMC-2026-10-28": "combo_ECB+FOMC", "ECB-2026-10-29": ""}


def test_per_event_factor_equals_type_factor():
    ev = (ScheduledEvent(0.01, "FOMC", "a"), ScheduledEvent(0.02, "ECB", "b"), ScheduledEvent(0.5, "CPI", "c"))
    a = PerEventJumps(ev, {"a": 0.006**2, "b": 0.005**2, "c": 0.004**2})
    b = ScheduledEventJumps(ev, {"FOMC": 0.006, "ECB": 0.005, "CPI": 0.004, "NFP": 0.0})
    u = np.linspace(-80, 80, 41) - 0.5j
    for T in (0.005, 0.015, 0.1, 1.0):
        np.testing.assert_allclose(a.cf_factor(u, T), b.cf_factor(u, T), rtol=1e-13)


def test_sparse_jacobian_pattern_is_complete():
    """Every nonzero of a dense finite-difference Jacobian lies inside the declared sparsity pattern."""
    panel, truth = _panel((FOMC, ECB))
    mapping, keys, _ = event_parameter_map(panel)
    spec = PooledSpec.make("HB-MJ", keys)
    prob = _Problem(panel, spec, Objective(), mapping, smooth=1e-4, fixed_shared=None, threads=1)
    z = prob.pack(true_shared(truth, keys), truth.daily)
    r0 = prob.residuals(z)
    S = prob.sparsity().toarray().astype(bool)
    for j in range(len(z)):
        dz = np.zeros_like(z)
        dz[j] = 1e-6
        col = (prob.residuals(z + dz) - r0) != 0
        assert not (col & ~S[:, j]).any(), prob.names()[j]


def test_pooled_recovers_shared_event_variances_noise_free():
    """Event mechanics only: structural parameters held at truth (a 5-date, 2-tenor panel cannot pin
    kappa/sigma/jumps down; the full study in scripts/pooled_study.py frees them)."""
    panel, truth = _panel((FOMC, ECB))
    mapping, keys, _ = event_parameter_map(panel)
    spec = PooledSpec.make("HB-MJ", keys)
    ts = true_shared(truth, keys)
    start = {**ts, "sigma_FOMC": 0.003, "sigma_ECB": 0.008}
    f = fit_pooled(panel, spec, Objective(), start, heuristic_local_start(panel, spec), mapping, threads=1,
                   fixed_shared=TRUTH_SHARED)
    assert f.success and f.iv_rmse < 1e-6
    assert f.shared["sigma_FOMC"] == pytest.approx(0.006, rel=1e-4)
    assert f.shared["sigma_ECB"] == pytest.approx(0.005, rel=1e-4)
    for est, tru in zip(f.local, truth.daily):
        assert est["v0"] == pytest.approx(tru["v0"], rel=1e-3)
    idf = pooled_identifiability(f)
    assert abs(idf["corr"].loc["sigma_FOMC", "sigma_ECB"]) < 0.9  # separated by the Wednesday valuation


def test_combined_parameter_recovers_summed_variance():
    panel, truth = _panel((FOMC, ECB_MOVED))
    mapping, keys, _ = event_parameter_map(panel)
    spec = PooledSpec.make("HB-MJ", keys)
    ts = true_shared(truth, keys)
    f = fit_pooled(panel, spec, Objective(), {**ts, "combo_ECB+FOMC": 0.004}, heuristic_local_start(panel, spec),
                   mapping, threads=1, fixed_shared=TRUTH_SHARED)
    assert f.shared["combo_ECB+FOMC"] ** 2 == pytest.approx(0.006**2 + 0.005**2, rel=1e-4)


def test_timeout_returns_best_point_flagged_not_converged():
    panel, truth = _panel((FOMC, ECB))
    mapping, keys, _ = event_parameter_map(panel)
    spec = PooledSpec.make("HB-MJ", keys)
    lines = []
    f = fit_pooled(panel, spec, Objective(), {**true_shared(truth, keys), "sigma_FOMC": 0.003},
                   heuristic_local_start(panel, spec), mapping, threads=1, timeout_s=1e-9, progress=lines.append)
    assert f.status == -99 and not f.success and "timed out" in f.message
    assert np.isfinite(f.cost) and any("TIMEOUT" in s for s in lines)


def test_leave_one_tenor_out_split_and_holdout_error():
    panel, truth = _panel((FOMC, ECB), tenors=("ON", "1W", "2W"))
    train, held = split_heldout(panel, "2W")
    assert all(set(d.data.labels) == {"ON", "1W"} for d in train.dates)
    assert all(set(h.labels) == {"2W"} for h in held)
    mapping, keys, _ = event_parameter_map(train)
    spec = PooledSpec.make("HB-MJ", keys)
    f = fit_pooled(train, spec, Objective(), true_shared(truth, keys), truth.daily, mapping, threads=1, max_nfev=5)
    oos = holdout_error(f, train, held)
    assert oos["oos_iv_rmse_volpts"] < 1e-4  # truth parameters predict the held-out tenor exactly

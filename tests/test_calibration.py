"""Calibration engine: transforms, weights, IV inversion, and small synthetic recoveries.

The full recovery study (more cases, more starts, noise) is scripts/calibration_recovery.py.
"""

import numpy as np
import pytest

from src.calibration import (
    BATES, HBMJ, HESTON, HESTON_PARAMS, JUMP_PARAMS, EVENT_PARAMS, CalibrationData, Objective, calibrate, evaluate,
    fit, fit_metrics, identifiability, subset_masks, synthetic_surface, variance_step, weights,
)
from src.events import ScheduledEvent
from src.models.black_scholes import GKParams, black76_price, implied_vol_array, implied_vol_fast
from src.models.hbmj import ScheduledEventJumps
from src.models.nested import NestedModel
from src.pricing_engine import price_european

DAY = 1 / 365
TRUE_H = dict(v0=0.0049, kappa=2.5, theta=0.0081, sigma=0.45, rho=-0.25)
JUMPS = dict(lam=1.0, mu_J=-0.02, sigma_J=0.015)
FOMC = ScheduledEvent(3.3 * DAY, "FOMC", "FOMC-1")
OBJ = Objective()


# ---------------------------------------------------------------- building blocks
def test_fast_iv_matches_brent_and_rejects_invalid():
    F, D = 1.16, 0.99
    K = F * np.exp(np.linspace(-0.2, 0.2, 21))
    T = np.full(21, 0.25)
    C = K >= F
    p = black76_price(F, K, T, 0.09, D, C)
    fast, ok = implied_vol_fast(p, F, K, T, D, C, guess=np.full(21, 0.2))
    slow, _ = implied_vol_array(p, F, K, T, D, C)
    assert ok.all()
    np.testing.assert_allclose(fast, slow, atol=1e-9)
    bad, ok_bad = implied_vol_fast(np.array([-1.0, 5.0, np.nan]), F, F, 0.25, D, True)
    assert not ok_bad.any() and np.isnan(bad).all()


@pytest.mark.parametrize("spec", HESTON_PARAMS + JUMP_PARAMS + EVENT_PARAMS, ids=lambda s: s.name)
def test_transforms_round_trip_and_starts_are_strictly_feasible(spec):
    lo, hi = spec.z_bounds()
    for x in np.linspace(spec.lower, spec.upper, 7)[1:-1]:
        assert spec.from_z(spec.to_z(x)) == pytest.approx(x, rel=1e-12, abs=1e-15)
    for x in (spec.lower, spec.upper, spec.lower - 1, spec.upper + 1):
        z = spec.to_z(spec.clip(x))
        assert lo < z < hi


def test_positive_and_correlation_bounds_are_respected():
    names = {p.name: p for p in HESTON_PARAMS + JUMP_PARAMS + EVENT_PARAMS}
    assert all(names[n].lower > 0 and names[n].transform == "log" for n in ("v0", "kappa", "theta", "sigma"))
    assert -1 < names["rho"].lower and names["rho"].upper < 1 and names["rho"].transform == "atanh"
    assert names["lam"].lower == 0 and names["sigma_J"].lower == 0          # can reach exactly 0
    assert all(names[f"sigma_{k}"].lower == 0 for k in ("FOMC", "ECB", "CPI", "NFP"))


def test_vega_weights_are_per_expiry_so_short_dated_options_keep_weight():
    data = synthetic_surface(HESTON.build(TRUE_H, ()), [7 * DAY, 365 * DAY])
    w = weights(data, Objective("iv", "vega"))
    short, long_ = data.contracts.T < 0.1, data.contracts.T > 0.9
    assert w[short].max() == pytest.approx(w[long_].max())          # ATM weight equal across expiries
    assert w[short].min() < 0.8 * w[short].max()                     # wings down-weighted
    assert weights(data, Objective("iv", "equal")) == pytest.approx(np.ones(len(w)))
    wb = weights(data, Objective("iv", "inv_bidask_var"))
    assert wb.mean() == pytest.approx(1.0)


# ---------------------------------------------------------------- recoveries (small)
def test_heston_recovery_from_different_starts():
    data = synthetic_surface(HESTON.build(TRUE_H, ()), [14 * DAY, 61 * DAY, 182 * DAY, 365 * DAY])
    starts = [dict(v0=0.004, kappa=0.5, theta=0.01, sigma=0.2, rho=0.3), dict(v0=0.02, kappa=5.0, theta=0.003, sigma=1.0, rho=-0.6)]
    ms = calibrate(HESTON, data, OBJ, starts)
    for f in ms.all:
        assert f.success and f.n_invalid == 0
        for k, v in TRUE_H.items():
            assert f.params[k] == pytest.approx(v, rel=1e-4), k
    assert ms.best.feller_ratio == pytest.approx(2 * 2.5 * 0.0081 / 0.45**2, rel=1e-4)


def test_bates_on_heston_surface_collapses_lambda_to_zero():
    data = synthetic_surface(HESTON.build(TRUE_H, ()), [3 * DAY, 14 * DAY, 61 * DAY, 182 * DAY])
    f = fit(BATES, data, OBJ, {**TRUE_H, "lam": 0.5, "mu_J": 0.0, "sigma_J": 0.01})
    # With no jumps in the data, the jump component must switch itself off: its variance rate
    # lam * (mu_J^2 + sigma_J^2) goes to ~0. lam alone need not reach 0 -- once the jump size is ~0,
    # lam is not identified (an identifiability fact, reported by the diagnostics).
    jump_var = f.params["lam"] * (f.params["mu_J"] ** 2 + f.params["sigma_J"] ** 2)
    assert jump_var < 1e-8
    assert f.iv_rmse < 1e-6
    for k, v in TRUE_H.items():
        assert f.params[k] == pytest.approx(v, rel=1e-3), k


def test_hbmj_with_zero_event_sigmas_prices_exactly_like_bates():
    x = {**TRUE_H, **JUMPS}
    ev = (FOMC, ScheduledEvent(10 * DAY, "ECB", "ECB-1"))
    hb = HBMJ.build({**x, "sigma_FOMC": 0, "sigma_ECB": 0, "sigma_CPI": 0, "sigma_NFP": 0}, ev)
    b = BATES.build(x, ev)
    K, T = np.array([1.14, 1.16, 1.18]), 14 * DAY
    np.testing.assert_array_equal(price_european(hb, 1.16, K, T, 0.999, True), price_european(b, 1.16, K, T, 0.999, True))


def test_event_sigma_recovered_from_straddling_expiries_and_absent_type_not_identified():
    events = (FOMC, ScheduledEvent(300 * DAY, "ECB", "ECB-late"))  # ECB beyond every expiry
    true = {**TRUE_H, **JUMPS, "sigma_FOMC": 0.006, "sigma_ECB": 0.005, "sigma_CPI": 0.0, "sigma_NFP": 0.0}
    data = synthetic_surface(HBMJ.build(true, events), [1 * DAY, 2.8 * DAY, 3.8 * DAY, 7 * DAY, 30 * DAY, 91 * DAY], events)
    assert data.identifiable_event_types() == ("FOMC",)
    base = {k: true[k] for k in (*TRUE_H, *JUMPS)}
    fixed = {**base, "sigma_ECB": 0.0, "sigma_CPI": 0.0, "sigma_NFP": 0.0}
    f = fit(HBMJ, data, OBJ, {**base, "sigma_FOMC": 0.002, "sigma_ECB": 0, "sigma_CPI": 0, "sigma_NFP": 0}, fixed=fixed)
    assert f.free == ("sigma_FOMC",)
    assert f.params["sigma_FOMC"] == pytest.approx(0.006, rel=1e-5)
    # The ATM total-variance step across FOMC also contains ~1 day of diffusion + Poisson-jump
    # variance. Differencing against the same surface without the event isolates sigma_FOMC^2.
    no_event = synthetic_surface(HBMJ.build({**true, "sigma_FOMC": 0.0}, events),
                                 [1 * DAY, 2.8 * DAY, 3.8 * DAY, 7 * DAY, 30 * DAY, 91 * DAY], events)
    step = variance_step(data, data.iv, FOMC) - variance_step(no_event, no_event.iv, FOMC)
    # ATM implied variance is only approximately additive when the base distribution is fat-tailed
    # (ATM prices track mean absolute deviation, not variance): ~3% here; exact for a Gaussian base
    # (test_variance_step_exact_under_gaussian_diffusion). The calibrated sigma above is exact.
    assert step == pytest.approx(0.006**2, rel=0.05)


def test_unspanned_event_parameter_is_flagged_by_identifiability():
    events = (FOMC, ScheduledEvent(300 * DAY, "ECB", "ECB-late"))
    true = {**TRUE_H, **JUMPS, "sigma_FOMC": 0.006, "sigma_ECB": 0.005, "sigma_CPI": 0.0, "sigma_NFP": 0.0}
    data = synthetic_surface(HBMJ.build(true, events), [2.8 * DAY, 3.8 * DAY, 30 * DAY], events)
    fixed = {k: true[k] for k in (*TRUE_H, *JUMPS)} | {"sigma_CPI": 0.0, "sigma_NFP": 0.0}
    f = fit(HBMJ, data, OBJ, true, fixed=fixed)  # deliberately leave sigma_ECB free
    idf = identifiability(f, data, OBJ)
    assert "sigma_ECB" in idf["zero_columns"]


def test_variance_step_exact_under_gaussian_diffusion():
    ev = (ScheduledEvent(5 * DAY, "CPI", "CPI-1"),)
    m = NestedModel(GKParams(0.07), events=ScheduledEventJumps(ev, {"FOMC": 0, "ECB": 0, "CPI": 0.005, "NFP": 0}))
    data = synthetic_surface(m, [4.99 * DAY, 5.01 * DAY], ev, z_grid=(0.0,))
    assert variance_step(data, data.iv, ev[0]) == pytest.approx(0.005**2 + 0.07**2 * 0.02 * DAY, rel=1e-6)


# ---------------------------------------------------------------- metrics / breakdowns
def test_fit_metrics_and_information_criteria():
    data = synthetic_surface(HESTON.build(TRUE_H, ()), [14 * DAY, 91 * DAY], noise_vol=0.001, seed=3)
    f = fit(HESTON, data, OBJ, TRUE_H)
    m = fit_metrics(f, data, OBJ)
    n, k = m["n"], m["k"]
    ev = evaluate(HESTON.build(f.params, ()), data, OBJ)
    rss = float(np.sum(ev.residuals**2))
    assert k == 5 and m["aic"] == pytest.approx(n * np.log(rss / n) + 2 * k)
    assert m["bic"] == pytest.approx(n * np.log(rss / n) + k * np.log(n))
    assert m["iv_rmse_volpts"] == pytest.approx(100 * np.sqrt(np.mean((ev.iv_model - data.iv) ** 2)))


def test_subset_masks_include_event_types_and_before_after():
    events = (FOMC,)
    true = {**TRUE_H, **JUMPS, "sigma_FOMC": 0.006, "sigma_ECB": 0, "sigma_CPI": 0, "sigma_NFP": 0}
    data = synthetic_surface(HBMJ.build(true, events), [2.8 * DAY, 3.8 * DAY, 30 * DAY], events)
    m = subset_masks(data)
    assert m["event_spanning"].sum() == 10 and m["non_event"].sum() == 5 and m["FOMC_spanning"].sum() == 10
    assert m["before_FOMC-1"].sum() == 5 and m["after_FOMC-1"].sum() == 5 and not m["ECB_spanning"].any()


def test_adjacent_events_without_separating_expiry_are_flagged_as_substitutes():
    """FOMC and ECB one day apart, no expiry in between: only sigma_FOMC^2 + sigma_ECB^2 is identified.
    The diagnostics must say so (ill-conditioned, correlation ~ -1), not declare one split correct."""
    ev = (ScheduledEvent(2.3 * DAY, "FOMC", "F"), ScheduledEvent(3.3 * DAY, "ECB", "E"))
    base = {**TRUE_H, **JUMPS}
    true = {**base, "sigma_FOMC": 0.006, "sigma_ECB": 0.005, "sigma_CPI": 0.0, "sigma_NFP": 0.0}
    data = synthetic_surface(HBMJ.build(true, ev), [1 * DAY, 7 * DAY, 30 * DAY], ev)
    fixed = {**base, "sigma_CPI": 0.0, "sigma_NFP": 0.0}
    a = fit(HBMJ, data, OBJ, {**true, "sigma_FOMC": 0.003, "sigma_ECB": 0.007}, fixed=fixed)
    b = fit(HBMJ, data, OBJ, {**true, "sigma_FOMC": 0.007, "sigma_ECB": 0.002}, fixed=fixed)
    total = lambda f: f.params["sigma_FOMC"] ** 2 + f.params["sigma_ECB"] ** 2
    assert total(a) == pytest.approx(0.006**2 + 0.005**2, rel=1e-6) == total(b)   # the sum is identified
    assert abs(a.params["sigma_FOMC"] - b.params["sigma_FOMC"]) > 5e-4              # the split is not
    idf = identifiability(a, data, OBJ)
    assert idf["corr"].loc["sigma_FOMC", "sigma_ECB"] < -0.99
    assert any("ill-conditioned" in f for f in idf["flags"])

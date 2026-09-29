"""HB-MJ = Bates x scheduled-time, stochastic-size event factor.

All model comparisons below use one shared, fixed set of contract coordinates.
"""

import numpy as np
import pytest

from src.metrics import ContractSet, evaluate_models
from src.models.bates import PoissonJumps, bates
from src.models.black_scholes import GKParams, black76_price, implied_vol_array
from src.models.hbmj import ScheduledEvent, ScheduledEventJumps, event_cf, hbmj
from src.models.heston import HestonParams
from src.models.nested import NestedModel, martingale_error
from src.pricing_engine import price_european, price_european_quad

HESTON = HestonParams(kappa=2.0, theta=0.0064, sigma=0.3, rho=-0.2, v0=0.0064)
JUMPS = PoissonJumps(lam=2.0, mu_J=-0.01, sigma_J=0.02)
SIGMAS = {"FOMC": 0.006, "ECB": 0.005, "CPI": 0.004, "NFP": 0.0035}
DAY = 1 / 365
F, D = 1.085, 0.99
T_1W = 7 * DAY
K_1W = F * np.exp(np.array([-2, -1, 0, 1, 2]) * 0.08 * np.sqrt(T_1W))  # fixed contracts, all models


def _events(*pairs):
    return tuple(ScheduledEvent(tau, kind) for tau, kind in pairs)


def _model(events, sigmas=SIGMAS):
    return hbmj(HESTON, JUMPS, ScheduledEventJumps(events, sigmas))


# 1 ---------------------------------------------------------------------------
def test_zero_event_sigmas_reproduce_bates_exactly():
    ev = _events((2 * DAY, "FOMC"), (3 * DAY, "CPI"), (5 * DAY, "NFP"), (6 * DAY, "ECB"))
    m = _model(ev, {k: 0.0 for k in SIGMAS})
    b = bates(HESTON, JUMPS.lam, JUMPS.mu_J, JUMPS.sigma_J)
    for T in (T_1W, 30 * DAY):
        np.testing.assert_array_equal(price_european(m, F, K_1W, T, D, K_1W >= F), price_european(b, F, K_1W, T, D, K_1W >= F))
    assert m.without("events") == b  # switching the component off gives the Bates object itself


# 2 ---------------------------------------------------------------------------
def test_events_after_expiry_or_already_announced_have_zero_effect():
    ev = _events((8 * DAY, "FOMC"), (30 * DAY, "ECB"), (0.0, "CPI"), (-DAY, "NFP"))
    m = _model(ev)
    b = m.without("events")
    assert m.events.events_before(T_1W) == ()
    np.testing.assert_array_equal(price_european(m, F, K_1W, T_1W, D, K_1W >= F), price_european(b, F, K_1W, T_1W, D, K_1W >= F))


def test_event_exactly_at_expiry_is_included():
    m = _model(_events((T_1W, "FOMC")))
    assert len(m.events.events_before(T_1W)) == 1


# 3 ---------------------------------------------------------------------------
def test_expiry_crossing_event_adds_exact_variance_under_gaussian_diffusion():
    """GK diffusion + one normal event: X_T stays normal, so implied total variance is
    exactly vol^2 T before the event and vol^2 T + sigma_E^2 after it, at every strike."""
    vol, tau, s = 0.08, 3 * DAY, SIGMAS["FOMC"]
    m = NestedModel(GKParams(vol), events=ScheduledEventJumps(_events((tau, "FOMC")), SIGMAS))
    for T, extra in ((tau - 1e-6, 0.0), (tau + 1e-6, s**2)):
        K = F * np.exp(np.linspace(-0.02, 0.02, 9))
        iv, _ = implied_vol_array(price_european(m, F, K, T, D, K >= F), F, K, T, D, K >= F)
        np.testing.assert_allclose(iv**2 * T, vol**2 * T + extra, rtol=1e-8)


def test_expiry_crossing_event_adds_event_factor_under_hbmj():
    """Full HB-MJ: moving expiry from just before to just after tau multiplies the CF by
    exactly the event factor (up to the vanishing diffusion over 2 eps), and ATM total
    implied variance jumps by ~sigma_E^2."""
    tau, s, eps = 3 * DAY, SIGMAS["FOMC"], 1e-9
    m = _model(_events((tau, "FOMC")))
    u = np.linspace(-200, 200, 81) - 0.5j
    ratio = m.cf(u, tau + eps) / m.cf(u, tau - eps)
    np.testing.assert_allclose(ratio, event_cf(u, s), rtol=1e-5)

    def atm_total_var(T):
        iv, _ = implied_vol_array(price_european(m, F, F, T, D, True), F, F, T, D, True)
        return iv[()] ** 2 * T

    step = atm_total_var(tau + 1e-7) - atm_total_var(tau - 1e-7)
    assert step == pytest.approx(s**2, rel=0.03)


# 4 ---------------------------------------------------------------------------
def test_event_factor_normalisation_and_martingale():
    ev = ScheduledEventJumps(_events((2 * DAY, "FOMC"), (5 * DAY, "NFP")), SIGMAS)
    for T in (DAY, T_1W, 1.0):
        assert abs(ev.cf_factor(np.array([0.0]), T)[0] - 1) < 1e-15
        assert martingale_error(ev, T) < 1e-15
        assert martingale_error(_model(ev.events), T) < 1e-13


# 5 ---------------------------------------------------------------------------
def test_two_events_multiply_their_factors():
    u = np.linspace(-100, 100, 101) + 0.25j
    a, b = ScheduledEvent(2 * DAY, "CPI"), ScheduledEvent(4 * DAY, "FOMC")
    both = ScheduledEventJumps((a, b), SIGMAS).cf_factor(u, T_1W)
    only_a = ScheduledEventJumps((a,), SIGMAS).cf_factor(u, T_1W)
    only_b = ScheduledEventJumps((b,), SIGMAS).cf_factor(u, T_1W)
    np.testing.assert_allclose(both, only_a * only_b, rtol=1e-14)
    # Sum of independent normals is normal: equivalently one jump with variance s_a^2 + s_b^2.
    np.testing.assert_allclose(both, event_cf(u, np.hypot(SIGMAS["CPI"], SIGMAS["FOMC"])), rtol=1e-12)


# 6 ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [T_1W, 14 * DAY, 30 * DAY, 0.25])
def test_integration_fft_and_reference_agree_with_events(T):
    ev = _events((2 * DAY, "FOMC"), (3 * DAY, "CPI"), (9 * DAY, "NFP"), (20 * DAY, "ECB"), (40 * DAY, "FOMC"))
    m = _model(ev)
    K = F * np.exp(np.linspace(-3, 3, 25) * 0.09 * np.sqrt(T))
    otm_call = K >= F
    direct = price_european(m, F, K, T, D, otm_call)
    fft = price_european(m, F, K, T, D, otm_call, method="fft")
    np.testing.assert_allclose(fft, direct, atol=5e-9 * F)
    for i in (0, 12, 24):
        assert direct[i] == pytest.approx(price_european_quad(m, F, K[i], T, D, otm_call[i]), abs=1e-10)


# 7 ---------------------------------------------------------------------------
def _fixed_contract_stats(sigma_fomc):
    m = _model(_events((3 * DAY, "FOMC")), {**SIGMAS, "FOMC": sigma_fomc})
    prices = price_european(m, F, K_1W, T_1W, D, K_1W >= F)
    iv, _ = implied_vol_array(prices, F, K_1W, T_1W, D, K_1W >= F)
    return iv, prices


def test_larger_event_sigma_raises_tail_value_and_atm_vol_of_spanning_option():
    prev_iv, prev_px = _fixed_contract_stats(0.0)
    for s in (0.0025, 0.005, 0.0075, 0.01):
        iv, px = _fixed_contract_stats(s)
        assert (px > prev_px).all()          # every contract, incl. both 2-sd wings, is worth more
        assert iv[2] > prev_iv[2]            # ATM vol up
        prev_iv, prev_px = iv, px


def test_gaussian_event_jump_adds_variance_but_not_kurtosis():
    """Documented property of the parsimonious spec: a normal event jump dilutes the fat tails
    that come from Heston/Bates, so implied-vol *curvature* on fixed contracts falls as sigma_E
    rises. (Under a pure GK diffusion it leaves the smile exactly flat -- see test 3.)"""
    bfs = []
    for s in (0.0, 0.0025, 0.005, 0.0075, 0.01):
        iv, _ = _fixed_contract_stats(s)
        bfs.append(0.5 * (iv[1] + iv[3]) - iv[2])
    assert all(a > b for a, b in zip(bfs, bfs[1:]))


# 8 ---------------------------------------------------------------------------
def test_bates_and_hbmj_compared_on_identical_contracts():
    T = np.repeat([T_1W, 30 * DAY], 5)
    K = np.tile(K_1W, 2)
    contracts = ContractSet.from_arrays(F, K, T, D, K >= F)
    hb = _model(_events((3 * DAY, "FOMC"), (10 * DAY, "CPI")))
    table = evaluate_models({"Bates": hb.without("events"), "HB-MJ": hb}, contracts)
    by_model = {name: g.sort_values("contract_id") for name, g in table.groupby("model")}
    for col in ("contract_id", "F", "K", "T", "D", "is_call"):
        np.testing.assert_array_equal(by_model["Bates"][col].to_numpy(), by_model["HB-MJ"][col].to_numpy())
    assert (table["iv_status"] == "ok").all()
    with pytest.raises(ValueError):
        contracts.K[0] = 1.0  # coordinates are read-only: no silent per-model rescaling


@pytest.mark.parametrize("T", [0.5 * DAY, 1 * DAY, 7 * DAY])
def test_integration_accurate_at_calibration_bound_extremes(T):
    """Guards the adaptive panel width of the integration engine: sharp CF features from large mean
    jumps (period 2 pi/|mu_J|), tiny jump dispersion and the largest allowed event sigma."""
    h = HestonParams(kappa=2.0, theta=0.0064, sigma=0.6, rho=-0.3, v0=0.0036)
    m = hbmj(h, PoissonJumps(lam=5.0, mu_J=-0.10, sigma_J=0.005),
             ScheduledEventJumps(_events((0.25 * DAY, "FOMC")), {**SIGMAS, "FOMC": 0.03}))
    K = F * np.exp(np.array([-0.15, -0.05, -0.01, 0.0, 0.01, 0.05]))
    for k in K:
        assert price_european(m, F, k, T, D, k >= F) == pytest.approx(
            price_european_quad(m, F, k, T, D, k >= F), abs=1e-10)

"""Bates (1996) = Heston x compensated compound-Poisson jump factor.

Verification plan (all before any calibration):
 1. lam = 0 reproduces Heston exactly.
 2. sigma_J = 0 reduces to fixed-size Poisson jumps (checked against an independent
    Merton-style Poisson-mixture formula, which also covers sigma_J > 0).
 3. phi(0) = 1 and phi(-i) = 1 (forward martingale).
 4. Direct integration, FFT and the Gil-Pelaez reference agree under Bates parameters.
 5. More jump intensity / dispersion -> more short-maturity smile curvature and tail value.
 6. Heston vs Bates with identical Heston parameters isolates the jump effect.
"""

import numpy as np
import pytest
from scipy.stats import poisson

from src.models.bates import PoissonJumps, bates
from src.models.black_scholes import GKParams, black76_price, implied_vol_array
from src.models.heston import HestonParams
from src.models.nested import NestedModel, martingale_error
from src.pricing_engine import price_european, price_european_quad

HESTON = HestonParams(kappa=2.0, theta=0.0064, sigma=0.3, rho=-0.2, v0=0.0064)
JUMPS = dict(lam=2.0, mu_J=-0.01, sigma_J=0.02)
F, D = 1.085, 0.99
T_1W, T_1M, T_1Y = 7 / 365, 30 / 365, 1.0


def _otm_iv(model, K, T):
    is_call = K >= F
    iv, status = implied_vol_array(price_european(model, F, K, T, D, is_call), F, K, T, D, is_call)
    assert (status == "ok").all()
    return iv


def _strikes(T, n_sd=(-2.0, -1.0, 0.0, 1.0, 2.0), vol=0.08):
    return F * np.exp(np.array(n_sd) * vol * np.sqrt(T))


def _butterfly(iv):
    """Wing average minus ATM for strikes at (-2, -1, 0, +1, +2) sd: a smile-curvature measure."""
    return 0.5 * (iv[0] + iv[-1]) - iv[2]


def merton_series_price(vol, lam, mu_J, sigma_J, K, T, is_call, n_max=80):
    """Independent check: condition on the number of jumps n (Merton 1976).

    Given n jumps, X_T is normal with variance vol^2 T + n sigma_J^2 and mean shifted by
    n mu_J - lam kbar T, i.e. a Black-76 price with an adjusted forward and vol. Weight by
    the Poisson probability of n jumps.
    """
    kbar = np.expm1(mu_J + 0.5 * sigma_J**2)
    total = 0.0
    for n in range(n_max):
        var_n = vol**2 * T + n * sigma_J**2
        F_n = F * np.exp(n * mu_J + 0.5 * n * sigma_J**2 - lam * kbar * T)  # E[S_T | n jumps]
        total = total + poisson.pmf(n, lam * T) * black76_price(F_n, K, T, np.sqrt(var_n / T), D, is_call)
    return total


# 1 ---------------------------------------------------------------------------
def test_zero_intensity_reproduces_heston_exactly():
    m = bates(HESTON, lam=0.0, mu_J=-0.05, sigma_J=0.1)  # jump sizes irrelevant when lam = 0
    K = F * np.exp(np.linspace(-0.08, 0.08, 17))
    for T in (T_1W, T_1M, T_1Y):
        np.testing.assert_array_equal(price_european(m, F, K, T, D, True), price_european(HESTON, F, K, T, D, True))
    u = np.linspace(-50, 50, 201) - 0.5j
    np.testing.assert_array_equal(m.cf(u, 0.3), HESTON.cf(u, 0.3))


# 2 ---------------------------------------------------------------------------
@pytest.mark.parametrize("sigma_J", [0.0, 0.03])
@pytest.mark.parametrize("T", [T_1W, T_1M, T_1Y])
def test_jump_factor_matches_merton_poisson_mixture(sigma_J, T):
    """GK diffusion x jump factor must equal the Poisson-weighted Black-76 sum.

    sigma_J = 0 is the fixed-size case: every jump is exactly e^{mu_J}.
    """
    lam, mu_J, vol = 3.0, -0.02, 0.07
    m = NestedModel(GKParams(vol), jumps=PoissonJumps(lam, mu_J, sigma_J))
    K = F * np.exp(np.linspace(-0.10, 0.06, 17))
    for is_call in (True, False):
        np.testing.assert_allclose(
            price_european(m, F, K, T, D, is_call),
            merton_series_price(vol, lam, mu_J, sigma_J, K, T, is_call),
            atol=1e-11,
        )


def test_fixed_size_jump_cf_closed_form():
    """sigma_J = 0: phi_J(u) = exp(lam T (e^{i u mu} - 1 - i u (e^mu - 1)))."""
    lam, mu, T = 1.5, 0.015, 0.4
    u = np.linspace(-30, 30, 121) + 0.3j
    expected = np.exp(lam * T * (np.exp(1j * u * mu) - 1 - 1j * u * np.expm1(mu)))
    np.testing.assert_allclose(PoissonJumps(lam, mu, 0.0).cf_factor(u, T), expected, rtol=1e-14)


# 3 ---------------------------------------------------------------------------
@pytest.mark.parametrize("lam, mu_J, sigma_J", [(2.0, -0.01, 0.02), (0.5, 0.05, 0.0), (20.0, -0.1, 0.08)])
def test_cf_normalisation_and_forward_martingale(lam, mu_J, sigma_J):
    m = bates(HESTON, lam, mu_J, sigma_J)
    for T in (1 / 365, T_1M, 5.0):
        assert abs(m.cf(np.array([0.0]), T)[0] - 1.0) < 1e-14
        assert martingale_error(m, T) < 1e-13
        assert martingale_error(m.jumps, T) < 1e-13  # the factor on its own, too


# 4 ---------------------------------------------------------------------------
@pytest.mark.parametrize("T", [1 / 365, T_1W, T_1M, 0.25, T_1Y, 5.0])
def test_integration_fft_and_reference_agree(T):
    m = bates(HESTON, **JUMPS)
    K = F * np.exp(np.linspace(-3, 3, 25) * 0.09 * np.sqrt(T))
    otm_call = K >= F
    direct = price_european(m, F, K, T, D, otm_call)
    fft = price_european(m, F, K, T, D, otm_call, method="fft")
    np.testing.assert_allclose(fft, direct, atol=5e-9 * F)
    for i in (0, 12, 24):
        assert direct[i] == pytest.approx(price_european_quad(m, F, K[i], T, D, otm_call[i]), abs=1e-10)


# 5 ---------------------------------------------------------------------------
# All model comparisons use the same fixed contract strikes (_strikes(T), i.e. +-1, +-2 sd of a
# common 8% reference vol), never strikes rescaled to each model's own vol.

def _bf_and_tails(model, T):
    K = _strikes(T)
    tails = price_european(model, F, K[[0, -1]], T, D, np.array([False, True]))
    return _butterfly(_otm_iv(model, K, T)), tails


def test_more_jump_intensity_or_dispersion_steepens_1w_smile():
    """For this parameterization at 1W, raising jump intensity or jump-size dispersion raises
    both smile curvature and wing prices on fixed contracts."""
    prev = _bf_and_tails(HESTON, T_1W)
    for lam in (0.5, 2.0, 8.0, 32.0):
        cur = _bf_and_tails(bates(HESTON, lam=lam, mu_J=0.0, sigma_J=0.02), T_1W)
        assert cur[0] > prev[0] and (cur[1] > prev[1]).all()
        prev = cur
    prev = _bf_and_tails(bates(HESTON, lam=2.0, mu_J=0.0, sigma_J=0.0), T_1W)  # zero-size jumps = Heston
    for sigma_J in (0.01, 0.02, 0.04):
        cur = _bf_and_tails(bates(HESTON, lam=2.0, mu_J=0.0, sigma_J=sigma_J), T_1W)
        assert cur[0] > prev[0] and (cur[1] > prev[1]).all()
        prev = cur


def test_many_small_jumps_average_out_at_long_maturity():
    """Documented limit of the mechanism: with many jumps before expiry, their sum is close to
    normal (CLT), so at 1Y jumps mainly raise ATM vol and *flatten* the Heston smile."""
    bf_heston = _bf_and_tails(HESTON, T_1Y)[0]
    bf_bates = _bf_and_tails(bates(HESTON, lam=32.0, mu_J=0.0, sigma_J=0.02), T_1Y)[0]
    assert bf_bates < bf_heston


# 6 ---------------------------------------------------------------------------
def test_jump_effect_isolated_and_concentrated_at_short_maturities():
    """Same Heston parameters and same contracts, jumps on vs off. For this parameterization
    the extra curvature from jumps is largest at 1W and declines monotonically with maturity
    (turning negative by 1Y, see the CLT test above)."""
    m = bates(HESTON, lam=2.0, mu_J=0.0, sigma_J=0.02)
    extra_bf = [_bf_and_tails(m, T)[0] - _bf_and_tails(HESTON, T)[0] for T in (T_1W, T_1M, 0.25, T_1Y)]
    assert extra_bf[0] > 0.003  # > 0.3 vol points at 1W
    assert all(a > b for a, b in zip(extra_bf, extra_bf[1:]))

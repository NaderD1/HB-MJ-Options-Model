"""Carr–Madan FFT must agree with the direct (Lewis) integration engine.

Tolerances are set from a measured accuracy study (FFT vs integration, 1 day to 10 years,
strikes out to +-3 standard deviations): worst case ~0.17 bp of implied vol, ~1.5e-9 in
price per unit of forward.
"""

import numpy as np
import pytest

from src.models.black_scholes import GKParams, black76_price, implied_vol_array
from src.models.heston import HestonParams
from src.pricing_engine import carr_madan_grid, price_european

EURUSD_LIKE = HestonParams(kappa=2.0, theta=0.0064, sigma=0.3, rho=-0.2, v0=0.0064)
FANG_OOSTERLEE = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)
MATURITIES = [1 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 10.0]


def _strikes(model, F, T, n_sd=3.0, n=41):
    sd = np.sqrt(model.expected_variance(T) * T) if hasattr(model, "expected_variance") else model.vol * np.sqrt(T)
    return F * np.exp(np.linspace(-n_sd, n_sd, n) * sd)


@pytest.mark.parametrize("T", MATURITIES)
def test_fft_matches_black76_closed_form(T):
    F, D, gk = 1.085, 0.99, GKParams(0.08)
    K = _strikes(gk, F, T)
    for is_call in (True, False):
        fft = price_european(gk, F, K, T, D, is_call, method="fft")
        np.testing.assert_allclose(fft, black76_price(F, K, T, 0.08, D, is_call), atol=5e-9 * F)


@pytest.mark.parametrize("T", MATURITIES)
@pytest.mark.parametrize("model, F", [(EURUSD_LIKE, 1.085), (FANG_OOSTERLEE, 100.0)])
def test_fft_matches_integration_heston(model, F, T):
    D = 0.99
    K = _strikes(model, F, T)
    otm_call = K >= F  # compare OTM options: that's what we calibrate on
    fft = price_european(model, F, K, T, D, otm_call, method="fft")
    direct = price_european(model, F, K, T, D, otm_call, method="integration")
    np.testing.assert_allclose(fft, direct, atol=5e-9 * F)
    iv_fft, _ = implied_vol_array(fft, F, K, T, D, otm_call)
    iv_direct, _ = implied_vol_array(direct, F, K, T, D, otm_call)
    assert np.nanmax(np.abs(iv_fft - iv_direct)) < 0.5e-4  # 0.5 bp of vol = 0.005 vol points


@pytest.mark.parametrize("T, reference", [(1.0, 5.785155434376197), (10.0, 22.31894579115449)])
def test_fft_fang_oosterlee_benchmark(T, reference):
    assert price_european(FANG_OOSTERLEE, 100.0, 100.0, T, 1.0, True, method="fft") == pytest.approx(
        reference, abs=1e-7
    )


def test_fft_rejects_damping_beyond_moment_explosion():
    """E[S_T^11] is infinite for these parameters at T=10, so alpha=10 is inadmissible."""
    with pytest.raises(ValueError, match="Moment"):
        carr_madan_grid(FANG_OOSTERLEE.cf, 10.0, alpha=10.0)


def test_fft_put_call_parity():
    F, D, T = 1.085, 0.99, 0.25
    K = _strikes(EURUSD_LIKE, F, T)
    c = price_european(EURUSD_LIKE, F, K, T, D, True, method="fft")
    p = price_european(EURUSD_LIKE, F, K, T, D, False, method="fft")
    np.testing.assert_allclose(c - p, D * (F - K), atol=1e-14)


def test_fft_high_vol_of_vol_short_expiry_regression():
    """Found by the calibration recovery study: sigma=0.9, T=1W needs a ~1M-point grid. When the
    grid was silently capped at 2^18 the FFT carried a constant -2.6e-5 price offset."""
    m = HestonParams(kappa=1.0, theta=0.0064, sigma=0.9, rho=0.3, v0=0.01)
    F, D, T = 1.16, 0.999, 7 / 365
    K = F * np.exp(np.linspace(-0.03, 0.03, 7))
    np.testing.assert_allclose(price_european(m, F, K, T, D, K >= F, method="fft"),
                               price_european(m, F, K, T, D, K >= F), atol=5e-9 * F)


def test_fft_refuses_to_silently_coarsen_its_grid():
    m = HestonParams(kappa=1.0, theta=0.0064, sigma=0.9, rho=0.3, v0=0.01)
    with pytest.raises(ValueError, match="n_max"):
        carr_madan_grid(m.cf, 7 / 365, n_max=2**18)

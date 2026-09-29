"""Heston: published benchmarks first, then limits and internal consistency."""

import numpy as np
import pytest

from src.models.black_scholes import black76_price, implied_vol_array
from src.models.heston import HestonParams
from src.pricing_engine import price_european, price_european_quad

# Fang & Oosterlee (2008), "A novel pricing method for European options based on
# Fourier-cosine series expansions", SIAM J. Sci. Comput. 31(2) -- Heston test case.
FO = HestonParams(kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711, v0=0.0175)


@pytest.mark.parametrize(
    "T, published, quantlib",
    [
        # The published T=1 value is accurate to ~2e-8; QuantLib's AnalyticHestonEngine and both
        # of our independent formulas agree with each other to ~1e-12 at 5.785155434376.
        (1.0, 5.785155450, 5.785155434376197),
        (10.0, 22.318945791474590, 22.31894579115449),  # long T: the case that breaks the original formula
    ],
)
def test_fang_oosterlee_benchmark(T, published, quantlib):
    """S=100, K=100, r=q=0 (so F=100, D=1)."""
    price = price_european(FO, 100.0, 100.0, T, 1.0, True)
    assert price == pytest.approx(published, abs=3e-8)
    assert price == pytest.approx(quantlib, abs=1e-10)


def test_martingale_condition():
    """phi(-i) = E[S_T/F] = 1 and phi(0) = 1 for any maturity."""
    for T in (1 / 365, 0.25, 5.0):
        np.testing.assert_allclose(FO.cf(np.array([0.0, -1j]), T), [1.0, 1.0], atol=1e-13)


@pytest.mark.parametrize("T", [7 / 365, 0.25, 2.0])
@pytest.mark.parametrize("K", [0.95, 1.08, 1.20])
def test_engine_matches_independent_quad_formula(T, K):
    """Lewis fast integration vs Gil-Pelaez adaptive quad (two different formulas)."""
    p = HestonParams(kappa=2.0, theta=0.008, sigma=0.35, rho=-0.3, v0=0.006)
    F, D = 1.085, 0.99
    for is_call in (True, False):
        fast = price_european(p, F, K, T, D, is_call)
        ref = price_european_quad(p.cf, F, K, T, D, is_call)
        assert fast == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("T", [7 / 365, 0.5, 3.0])
def test_small_vol_of_vol_converges_to_black_scholes(T):
    """sigma -> 0: variance follows a deterministic path, so Heston = Black-76 with
    vol^2 = average of that path = HestonParams.expected_variance(T).

    The gap is O(rho * sigma) -- a real first-order skew effect, not numerical error -- so we
    check that it shrinks 10x each time sigma does.
    """
    F, D = 1.085, 0.99
    K = F * np.exp(np.linspace(-0.2, 0.2, 9) * np.sqrt(T))
    gaps = []
    for s in (1e-4, 1e-5, 1e-6, 1e-7):
        p = HestonParams(kappa=3.0, theta=0.01, sigma=s, rho=-0.5, v0=0.004)
        heston = price_european(p, F, K, T, D, True)
        bs = black76_price(F, K, T, np.sqrt(p.expected_variance(T)), D, True)
        gaps.append(np.max(np.abs(heston - bs)))
    assert gaps[-1] < 1e-8
    assert all(g2 < 0.2 * g1 for g1, g2 in zip(gaps, gaps[1:]))


def test_negative_rho_produces_downside_skew():
    """rho < 0: spot falls when vol rises -> low strikes get higher implied vol than high strikes."""
    F, D, T = 1.085, 0.99, 0.25
    K = np.array([F * 0.95, F * 1.05])
    for rho, sign in ((-0.5, 1), (0.5, -1)):
        p = HestonParams(kappa=2.0, theta=0.008, sigma=0.4, rho=rho, v0=0.008)
        prices = price_european(p, F, K, T, D, K > F)  # OTM options
        iv, _ = implied_vol_array(prices, F, K, T, D, K > F)
        assert sign * (iv[0] - iv[1]) > 0.001

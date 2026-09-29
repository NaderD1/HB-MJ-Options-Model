"""The engine must reproduce closed-form Black-76 before we trust it with anything else."""

import numpy as np
import pytest

from src.models.black_scholes import GKParams, black76_price
from src.pricing_engine import price_european, price_european_quad


@pytest.mark.parametrize("T", [2 / 365, 7 / 365, 30 / 365, 0.5, 2.0])
@pytest.mark.parametrize("vol", [0.05, 0.10, 0.30])
def test_engine_matches_black76(T, vol):
    F, D = 1.085, 0.99
    K = F * np.exp(np.linspace(-4, 4, 17) * vol * np.sqrt(T))  # +-4 standard deviations
    for is_call in (True, False):
        engine = price_european(GKParams(vol), F, K, T, D, is_call)
        exact = black76_price(F, K, T, vol, D, is_call)
        np.testing.assert_allclose(engine, exact, atol=1e-11)


def test_quad_reference_matches_black76():
    F, K, T, D, vol = 1.085, 1.10, 0.25, 0.99, 0.08
    for is_call in (True, False):
        assert price_european_quad(GKParams(vol), F, K, T, D, is_call) == pytest.approx(
            float(black76_price(F, K, T, vol, D, is_call)), abs=1e-10
        )


def test_vectorised_mixed_maturities():
    """One call with several maturities/strikes must equal pricing each one separately."""
    cf = GKParams(0.09)
    F, D = 1.08, 0.995
    K = np.array([1.00, 1.08, 1.15, 1.05])
    T = np.array([0.1, 0.1, 0.5, 1.0])
    C = np.array([True, False, True, False])
    together = price_european(cf, F, K, T, D, C)
    separate = [price_european(cf, F, k, t, D, c)[()] for k, t, c in zip(K, T, C)]
    np.testing.assert_allclose(together, separate, atol=1e-14)

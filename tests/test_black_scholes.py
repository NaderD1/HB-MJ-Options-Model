"""Tests for Garman–Kohlhagen pricing and the implied-vol solver."""

import numpy as np
import pytest

from src.models.black_scholes import (
    black76_price,
    black76_spot_delta,
    black76_vega,
    gk_price,
    implied_vol,
    implied_vol_array,
)


def test_textbook_value_hull():
    """Hull, Options Futures & Other Derivatives, Example 15.6 (r_f = 0 reduces GK to Black–Scholes).

    S=42, K=40, r=10%, vol=20%, T=0.5 -> call 4.76, put 0.81.
    """
    call = gk_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, True)
    put = gk_price(42.0, 40.0, 0.5, 0.10, 0.0, 0.20, False)
    assert call == pytest.approx(4.76, abs=5e-3)
    assert put == pytest.approx(0.81, abs=5e-3)


@pytest.mark.parametrize("K", [0.95, 1.05, 1.10, 1.20])
@pytest.mark.parametrize("T", [7 / 365, 0.25, 1.0])
def test_put_call_parity_fx(K, T):
    """C - P = D_d F - D_d K = S e^{-r_f T} - K e^{-r_d T}."""
    S, r_d, r_f, vol = 1.08, 0.045, 0.025, 0.08
    c = gk_price(S, K, T, r_d, r_f, vol, True)
    p = gk_price(S, K, T, r_d, r_f, vol, False)
    assert c - p == pytest.approx(S * np.exp(-r_f * T) - K * np.exp(-r_d * T), abs=1e-12)


def test_vega_matches_finite_difference():
    F, K, T, v, D = 1.09, 1.12, 0.3, 0.09, 0.98
    h = 1e-5
    fd = (black76_price(F, K, T, v + h, D, True) - black76_price(F, K, T, v - h, D, True)) / (2 * h)
    assert black76_vega(F, K, T, v, D) == pytest.approx(fd, rel=1e-6)


def test_spot_delta_matches_finite_difference():
    S, K, T, r_d, r_f, v = 1.08, 1.10, 0.25, 0.045, 0.025, 0.08
    h = 1e-6
    fd = (gk_price(S + h, K, T, r_d, r_f, v, True) - gk_price(S - h, K, T, r_d, r_f, v, True)) / (2 * h)
    F = S * np.exp((r_d - r_f) * T)
    assert black76_spot_delta(F, K, T, v, np.exp(-r_f * T), True) == pytest.approx(fd, rel=1e-6)


def test_iv_round_trip_grid():
    """Price with a known vol, invert, recover it -- across moneyness, maturity, calls and puts."""
    F, D = 1.085, 0.99
    Ks = F * np.exp(np.linspace(-0.25, 0.25, 11))
    Ts = np.array([2 / 365, 7 / 365, 30 / 365, 0.5, 1.0, 2.0])
    vols_true = np.array([0.03, 0.08, 0.15, 0.40, 1.2])
    K, T, V, C = np.meshgrid(Ks, Ts, vols_true, [True, False], indexing="ij")
    prices = black76_price(F, K, T, V, D, C)
    iv, status = implied_vol_array(prices, F, K, T, D, C)
    # Vol information lives in *time value* (price - intrinsic). Deep OTM *and* deep ITM options at
    # low vol/short T have ~0 time value and legitimately carry no vol info -- which is why we
    # calibrate on OTM options only.
    intrinsic = D * np.where(C, np.maximum(F - K, 0), np.maximum(K - F, 0))
    identifiable = prices - intrinsic > 1e-10
    assert (status[identifiable] == "ok").all()
    np.testing.assert_allclose(iv[identifiable], V[identifiable], atol=1e-6)


@pytest.mark.parametrize(
    "price, expected",
    [
        (0.0005, "below_intrinsic"),   # call with F-K = 0.05 in the money is worth at least ~0.0495
        (5.0, "above_upper_bound"),    # a call can never exceed D*F
        (-0.01, "invalid_input"),
        (np.nan, "invalid_input"),
    ],
)
def test_iv_failure_handling(price, expected):
    res = implied_vol(price, F=1.10, K=1.05, T=0.25, D=0.99, is_call=True)
    assert res.status == expected
    assert np.isnan(res.vol)

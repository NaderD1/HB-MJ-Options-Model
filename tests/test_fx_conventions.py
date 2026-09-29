"""FX delta conventions, ATM definitions, and ATM/RR/BF <-> strike-vol reconstruction."""

import numpy as np
import pytest

from src.fx_conventions import (
    FXConventions, atm_strike, build_smile, fx_delta, quotes_from_smile, strike_from_delta,
)

S = 1.16


def _market(T):
    D_d = np.exp(-0.038 * T)
    F = S * np.exp(0.017 * T)
    return F, D_d, F * D_d / S


@pytest.mark.parametrize("delta_type", ["spot", "forward", "spot_pa", "forward_pa"])
@pytest.mark.parametrize("T", [1 / 365, 7 / 365, 0.25, 1.0])
@pytest.mark.parametrize("omega, delta", [(1, 0.25), (1, 0.10), (-1, -0.25), (-1, -0.10)])
def test_delta_to_strike_round_trip(delta_type, T, omega, delta):
    F, _, D_f = _market(T)
    K = strike_from_delta(delta, F, T, 0.08, D_f, omega, delta_type)
    assert fx_delta(F, K, T, 0.08, D_f, omega, delta_type) == pytest.approx(delta, abs=1e-12)


def test_premium_adjusted_call_strike_is_on_the_quoted_branch():
    """PA call delta is not monotone; the market strike lies above the delta-maximising strike."""
    F, _, D_f = _market(1.0)
    vol = 0.30
    K = strike_from_delta(0.25, F, 1.0, vol, D_f, 1, "spot_pa")
    bump = fx_delta(F, K * 1.001, 1.0, vol, D_f, 1, "spot_pa")
    assert bump < 0.25  # delta decreasing at the solution -> right-hand branch


@pytest.mark.parametrize("premium_adjusted", [False, True])
def test_dns_atm_has_zero_straddle_delta(premium_adjusted):
    T, vol = 0.25, 0.08
    F, _, D_f = _market(T)
    dt = "spot_pa" if premium_adjusted else "spot"
    K = atm_strike(F, S, T, vol, "dns", premium_adjusted)
    assert fx_delta(F, K, T, vol, D_f, 1, dt) + fx_delta(F, K, T, vol, D_f, -1, dt) == pytest.approx(0, abs=1e-14)


def test_smile_strangle_formula():
    T = 30 / 365
    F, D_d, D_f = _market(T)
    conv = FXConventions("spot", "dns", "smile", "USD")
    pts = {p.bucket: p for p in build_smile(0.07, {0.25: (-0.004, 0.002)}, F, S, T, D_d, D_f, conv)}
    assert pts["25DC"].vol == pytest.approx(0.07 + 0.002 - 0.002)
    assert pts["25DP"].vol == pytest.approx(0.07 + 0.002 + 0.002)


@pytest.mark.parametrize("bf_type", ["smile", "market"])
@pytest.mark.parametrize("delta_type, prem", [("spot", "USD"), ("forward", "USD"), ("spot_pa", "EUR")])
@pytest.mark.parametrize("with_10d", [False, True])
@pytest.mark.parametrize("T", [1 / 365, 7 / 365, 0.25, 1.0])
def test_quotes_round_trip(bf_type, delta_type, prem, with_10d, T):
    F, D_d, D_f = _market(T)
    conv = FXConventions(delta_type, "dns", bf_type, prem)
    quotes = {0.25: (-0.0035, 0.0018)}
    if with_10d:
        quotes[0.10] = (-0.0065, 0.0060)
    pts = build_smile(0.071, quotes, F, S, T, D_d, D_f, conv)
    back = quotes_from_smile(pts, F, S, T, D_d, D_f, conv)
    assert back["atm"] == pytest.approx(0.071, abs=1e-12)
    for d, tag in ((0.25, "25"), (0.10, "10")):
        if d in quotes:
            assert back[f"rr{tag}"] == pytest.approx(quotes[d][0], abs=1e-12)
            assert back[f"bf{tag}"] == pytest.approx(quotes[d][1], abs=1e-11)
    for p in pts:
        if p.bucket != "ATM":
            assert fx_delta(F, p.strike, T, p.vol, D_f, 1 if p.is_call else -1, delta_type) == pytest.approx(p.delta, abs=1e-12)


def test_market_strangle_differs_from_smile_strangle():
    """Same BF number, different convention -> different wing vols (why the convention must be known)."""
    T = 0.25
    F, D_d, D_f = _market(T)
    q = {0.25: (-0.004, 0.002), 0.10: (-0.0075, 0.007)}
    sm = {p.bucket: p.vol for p in build_smile(0.07, q, F, S, T, D_d, D_f, FXConventions("spot", "dns", "smile", "USD"))}
    ms = {p.bucket: p.vol for p in build_smile(0.07, q, F, S, T, D_d, D_f, FXConventions("spot", "dns", "market", "USD"))}
    assert abs(sm["10DP"] - ms["10DP"]) > 1e-4

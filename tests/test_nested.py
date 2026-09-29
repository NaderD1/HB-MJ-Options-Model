"""The nested-model wrapper must reproduce each standalone model exactly when components are off."""

from dataclasses import dataclass

import numpy as np

from src.models.black_scholes import GKParams, black76_price
from src.models.heston import HestonParams
from src.models.nested import NestedModel, martingale_error
from src.pricing_engine import price_european

HESTON = HestonParams(kappa=2.0, theta=0.0064, sigma=0.3, rho=-0.2, v0=0.0064)
F, D = 1.085, 0.99
K = F * np.exp(np.linspace(-0.05, 0.05, 11))


@dataclass(frozen=True)
class _IdentityFactor:
    """Placeholder component (factor == 1) until Bates/HB-MJ factors exist."""

    def cf_factor(self, u, T):
        return np.ones_like(np.asarray(u, dtype=complex))


def test_nested_gk_equals_closed_form():
    m = NestedModel(GKParams(0.08))
    assert m.name == "GK"
    np.testing.assert_allclose(price_european(m, F, K, 0.25, D, True), black76_price(F, K, 0.25, 0.08, D, True), atol=1e-11)


def test_nested_heston_equals_standalone_heston_bit_for_bit():
    m = NestedModel(HESTON)
    assert m.name == "Heston"
    for T in (7 / 365, 0.25, 1.0):
        np.testing.assert_array_equal(price_european(m, F, K, T, D, True), price_european(HESTON, F, K, T, D, True))


def test_components_switch_on_and_off():
    full = NestedModel(HESTON, jumps=_IdentityFactor(), events=_IdentityFactor())
    assert full.name == "HB-MJ"
    assert full.without("events").name == "Bates"
    assert full.without("events", "jumps").name == "Heston"
    assert full.without("jumps").name == "Heston+events"
    u = np.linspace(-20, 20, 101) - 0.5j
    np.testing.assert_array_equal(full.cf(u, 0.25), HESTON.cf(u, 0.25))


def test_martingale_condition_holds():
    for m in (NestedModel(GKParams(0.1)), NestedModel(HESTON, jumps=_IdentityFactor())):
        for T in (1 / 365, 0.5, 5.0):
            assert martingale_error(m, T) < 1e-14

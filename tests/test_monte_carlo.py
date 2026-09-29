"""Monte Carlo (independent engine) vs characteristic-function prices.

Seeds are fixed, so these tests are deterministic. The acceptance band is |z| < 4 standard
errors per contract; the full grid (more paths, 1Y maturities, discretization study) lives
in scripts/mc_validation.py.
"""

import numpy as np
import pytest

from src.models.bates import PoissonJumps
from src.models.black_scholes import GKParams
from src.models.hbmj import ScheduledEvent, ScheduledEventJumps
from src.models.heston import HestonParams
from src.models.nested import NestedModel
from src.monte_carlo import _time_grid, mc_price, simulate_log_returns
from src.pricing_engine import price_european

HESTON = HestonParams(kappa=2.0, theta=0.0064, sigma=0.3, rho=-0.2, v0=0.0064)
JUMPS = PoissonJumps(2.0, -0.01, 0.02)
SIGMAS = {"FOMC": 0.006, "ECB": 0.005, "CPI": 0.004, "NFP": 0.0035}
DAY = 1 / 365
F, D = 1.085, 0.99
EVENTS = (ScheduledEvent(2 * DAY, "CPI"), ScheduledEvent(4 * DAY, "FOMC"),
          ScheduledEvent(9 * DAY, "NFP"), ScheduledEvent(16 * DAY, "ECB"))


def _check(model, T, n_pairs=60_000, steps_per_year=365 * 2, seed=7, max_z=4.0):
    K = F * np.exp(np.array([-2, -1, 0, 1, 2]) * 0.08 * np.sqrt(T))
    is_call = K >= F
    mc = mc_price(model, F, K, T, D, is_call, n_pairs, steps_per_year, seed)
    cf = price_european(model, F, K, T, D, is_call)
    z = (mc.price - cf) / mc.se
    assert np.all(np.abs(z) < max_z), f"z-scores {np.round(z, 2)}"
    assert abs(mc.martingale / mc.martingale_se) < max_z


@pytest.mark.parametrize(
    "model, T",
    [
        (NestedModel(HESTON), 30 * DAY),
        (NestedModel(HESTON), 0.25),
        (NestedModel(HESTON, JUMPS), 7 * DAY),
        (NestedModel(HESTON, JUMPS), 30 * DAY),
        (NestedModel(HESTON, PoissonJumps(0.5, -0.05, 0.05)), 30 * DAY),
        (NestedModel(HESTON, JUMPS, ScheduledEventJumps(EVENTS, SIGMAS)), 7 * DAY),
        (NestedModel(HESTON, JUMPS, ScheduledEventJumps(EVENTS, SIGMAS)), 30 * DAY),
    ],
    ids=["heston-1M", "heston-3M", "bates-1W", "bates-1M", "bates-bigjumps-1M", "hbmj-1W", "hbmj-1M"],
)
def test_mc_matches_cf(model, T):
    _check(model, T)


def test_exact_scheme_gk_plus_events():
    """GK diffusion + Gaussian events is simulated *exactly* (no discretization error),
    so this isolates the event-timing rule and compensator from Euler bias."""
    _check(NestedModel(GKParams(0.08), events=ScheduledEventJumps(EVENTS, SIGMAS)), 10 * DAY, steps_per_year=12)


def test_events_after_expiry_leave_paths_identical_to_bates():
    """No event before T -> no extra random draws -> the very same paths as Bates."""
    late = ScheduledEventJumps((ScheduledEvent(8 * DAY, "FOMC"),), SIGMAS)
    a = simulate_log_returns(NestedModel(HESTON, JUMPS, late), 7 * DAY, 1000, seed=3)
    b = simulate_log_returns(NestedModel(HESTON, JUMPS), 7 * DAY, 1000, seed=3)
    np.testing.assert_array_equal(a, b)


def test_event_times_are_grid_nodes():
    grid = _time_grid(0.1, 52, [0.013, 0.05, 0.2, -0.01])
    assert 0.013 in grid and 0.05 in grid
    assert grid.max() == 0.1 and grid.min() == 0.0  # events outside (0, T] are not added


def test_antithetic_pairs_are_mirrored():
    """GK: the Brownian parts of the pair are exact negatives, so X+ + X- = -vol^2 T."""
    X = simulate_log_returns(NestedModel(GKParams(0.1)), 0.5, 500, steps_per_year=10, seed=1)
    np.testing.assert_allclose(X[0] + X[1], -0.1**2 * 0.5, atol=1e-12)

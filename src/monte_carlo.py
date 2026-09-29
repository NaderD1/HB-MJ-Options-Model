"""Monte Carlo simulation of the nested models -- an independent *validation* engine.

Nothing here reuses the characteristic-function code: paths are generated from the model
dynamics directly, so agreement with the CF prices checks the CFs, the compensators and
the event-timing rule all at once. MC is never used for calibration.

Simulated state: X_t = ln(S_t / F_0(t))-type log-return relative to the forward, so the
risk-neutral drift contains only convexity and jump-compensator terms (no rates).

Heston variance: full-truncation Euler (Lord, Koekkoek & van Dijk 2010)
    v+     = max(v, 0)
    X     += -1/2 v+ dt + sqrt(v+ dt) Z1
    v     += kappa (theta - v+) dt + sigma sqrt(v+ dt) Z2,   corr(Z1, Z2) = rho
Negative v is allowed to persist in the state but is floored at 0 wherever it is used.
This is the least-biased of the simple Euler fixes, but it is *not* exact: the scheme has a
discretization bias of order dt that is largest when the Feller condition is badly
violated (variance often near 0) and when vol-of-vol is large. We measure that bias
directly by halving dt (see ``discretization_study``).

Poisson jumps (Bates): per step, N ~ Poisson(lam dt) jumps, whose summed log-size given N
is exactly N(N mu_J, N sigma_J^2); drift is reduced by lam * kbar * dt (compensator).
This part is exact given the grid -- no discretization error.

Scheduled events (HB-MJ): at the grid step containing tau_i, add
Y_i ~ N(-sigma_E^2/2, sigma_E^2). Also exact. The grid always has a node at every event
time inside (0, T], so the jump lands on the correct date (needed later for hedging).

Variance reduction: antithetic variates -- every normal draw is used with both signs
(Poisson counts are shared by the pair). Standard errors are computed from pair averages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.models.black_scholes import GKParams
from src.models.heston import HestonParams
from src.models.nested import NestedModel


def _time_grid(T: float, steps_per_year: int, event_times: list[float]) -> np.ndarray:
    """Uniform grid on [0, T] with extra nodes inserted at every event time in (0, T]."""
    n = max(int(np.ceil(T * steps_per_year)), 1)
    grid = np.linspace(0.0, T, n + 1)
    extra = [t for t in event_times if 0.0 < t <= T]
    return np.unique(np.concatenate([grid, extra]))


def simulate_log_returns(
    model: NestedModel,
    T: float,
    n_pairs: int,
    steps_per_year: int = 365 * 4,
    seed: int = 0,
) -> np.ndarray:
    """Simulate X_T = ln(S_T/F) for 2*n_pairs antithetic paths. Returns shape (2, n_pairs)."""
    rng = np.random.default_rng(seed)
    events = model.events.events_before(T) if model.events is not None else ()
    grid = _time_grid(T, steps_per_year, [e.tau for e in events])

    X = np.zeros((2, n_pairs))
    sign = np.array([[1.0], [-1.0]])  # antithetic pair: +Z and -Z
    diff = model.diffusion
    if isinstance(diff, HestonParams):
        v = np.full((2, n_pairs), diff.v0)
        rho_c = np.sqrt(1.0 - diff.rho**2)
    jumps = model.jumps

    for t0, t1 in zip(grid[:-1], grid[1:]):
        dt = t1 - t0
        z1 = sign * rng.standard_normal(n_pairs)
        if isinstance(diff, GKParams):
            X += -0.5 * diff.vol**2 * dt + diff.vol * np.sqrt(dt) * z1  # exact
        elif isinstance(diff, HestonParams):
            z2 = diff.rho * z1 + rho_c * sign * rng.standard_normal(n_pairs)
            v_pos = np.maximum(v, 0.0)
            sq = np.sqrt(v_pos * dt)
            X += -0.5 * v_pos * dt + sq * z1
            v += diff.kappa * (diff.theta - v_pos) * dt + diff.sigma * sq * z2
        else:
            raise TypeError(f"Unsupported diffusion {type(diff).__name__}")

        if jumps is not None and jumps.lam > 0:
            n_jumps = rng.poisson(jumps.lam * dt, n_pairs)  # shared by the antithetic pair
            zj = sign * rng.standard_normal(n_pairs)
            X += n_jumps * jumps.mu_J + np.sqrt(n_jumps) * jumps.sigma_J * zj - jumps.lam * jumps.kbar * dt

        for e in events:
            if t0 < e.tau <= t1:
                s = model.events.sigma_E[e.kind]
                X += -0.5 * s**2 + s * sign * rng.standard_normal(n_pairs)
    return X


@dataclass(frozen=True)
class MCPrices:
    price: np.ndarray        # MC estimate per strike
    se: np.ndarray           # standard error per strike
    martingale: float        # mean(S_T/F) - 1 (should be ~0 up to MC error + discretization bias)
    martingale_se: float


def mc_price(
    model: NestedModel,
    F: float,
    K: ArrayLike,
    T: float,
    D: float,
    is_call: ArrayLike,
    n_pairs: int = 100_000,
    steps_per_year: int = 365 * 4,
    seed: int = 0,
) -> MCPrices:
    """European prices by Monte Carlo on one set of paths for all strikes at maturity T."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), K.shape)
    X = simulate_log_returns(model, T, n_pairs, steps_per_year, seed)
    ST = F * np.exp(X)  # (2, n_pairs)
    payoff = np.where(
        is_call[:, None, None], np.maximum(ST[None] - K[:, None, None], 0.0), np.maximum(K[:, None, None] - ST[None], 0.0)
    )
    pair_mean = D * payoff.mean(axis=1)  # average the antithetic pair first -> iid samples
    growth = np.exp(X).mean(axis=0)
    return MCPrices(
        price=pair_mean.mean(axis=1),
        se=pair_mean.std(axis=1, ddof=1) / np.sqrt(n_pairs),
        martingale=float(growth.mean() - 1.0),
        martingale_se=float(growth.std(ddof=1) / np.sqrt(n_pairs)),
    )

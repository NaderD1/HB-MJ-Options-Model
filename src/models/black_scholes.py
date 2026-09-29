"""Garman–Kohlhagen (1983) FX option pricing and an implied-volatility solver.

Everything is written in *forward* form (Black-76):

    call = D * [F N(d1) - K N(d2)],   put = D * [K N(-d2) - F N(-d1)]
    d1 = [ln(F/K) + 0.5 vol^2 T] / (vol sqrt(T)),   d2 = d1 - vol sqrt(T)

with D = exp(-r_d T) the domestic (USD) discount factor and
F = S exp((r_d - r_f) T) the outright forward (covered interest parity).
Garman–Kohlhagen is exactly this with F and D built from spot and the two rates,
so every later model can share the same (F, D) inputs.

Constant volatility means one number fits every strike: a flat smile. That is
the benchmark every other model in this project must beat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq
from scipy.stats import norm

VOL_LOWER = 1e-6
VOL_UPPER = 5.0


def forward_from_rates(spot: float, T: float, r_d: float, r_f: float) -> float:
    """Outright forward under covered interest parity: F = S exp((r_d - r_f) T)."""
    return spot * np.exp((r_d - r_f) * T)


def black76_price(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, D: ArrayLike, is_call: ArrayLike
) -> np.ndarray:
    """Black-76 price of a European option on a forward (vectorised)."""
    F, K, T, vol, D = (np.asarray(x, dtype=float) for x in (F, K, T, vol, D))
    is_call = np.asarray(is_call, dtype=bool)
    sd = vol * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    d2 = d1 - sd
    call = D * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = D * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    # Zero-variance limit: discounted intrinsic value.
    call = np.where(sd > 0, call, D * np.maximum(F - K, 0.0))
    put = np.where(sd > 0, put, D * np.maximum(K - F, 0.0))
    return np.where(is_call, call, put)


def gk_price(
    spot: ArrayLike, K: ArrayLike, T: ArrayLike, r_d: ArrayLike, r_f: ArrayLike, vol: ArrayLike,
    is_call: ArrayLike,
) -> np.ndarray:
    """Garman–Kohlhagen price from spot and the domestic/foreign continuous rates."""
    spot, T, r_d, r_f = (np.asarray(x, dtype=float) for x in (spot, T, r_d, r_f))
    F = spot * np.exp((r_d - r_f) * T)
    D = np.exp(-r_d * T)
    return black76_price(F, K, T, vol, D, is_call)


def black76_vega(F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, D: ArrayLike) -> np.ndarray:
    """dPrice/dVol (per 1.00 of vol, i.e. divide by 100 for 'per vol point')."""
    F, K, T, vol, D = (np.asarray(x, dtype=float) for x in (F, K, T, vol, D))
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return D * F * norm.pdf(d1) * np.sqrt(T)


def black76_spot_delta(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, D_f: ArrayLike, is_call: ArrayLike
) -> np.ndarray:
    """Premium-unadjusted *spot* delta: dPrice/dSpot = D_f * N(d1) (call) or -D_f * N(-d1) (put).

    D_f = exp(-r_f T) is the foreign (EUR) discount factor. This is the delta
    convention used for EUR/USD quotes up to 1Y, which we need later to turn
    Bloomberg's 25-delta risk reversals/butterflies into strikes.
    """
    F, K, T, vol, D_f = (np.asarray(x, dtype=float) for x in (F, K, T, vol, D_f))
    is_call = np.asarray(is_call, dtype=bool)
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd**2) / sd
    return np.where(is_call, D_f * norm.cdf(d1), -D_f * norm.cdf(-d1))


# --------------------------------------------------------------------------- #
# Implied volatility
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IVResult:
    """Implied vol plus a status explaining any failure (vol is NaN unless status == 'ok')."""

    vol: float
    status: str  # ok | invalid_input | below_intrinsic | above_upper_bound | no_convergence


def implied_vol(
    price: float, F: float, K: float, T: float, D: float, is_call: bool, tol: float = 1e-10
) -> IVResult:
    """Invert Black-76 for volatility with Brent's method.

    Black-76 price is strictly increasing in vol between two no-arbitrage bounds:
        lower = D * max(F - K, 0)   (call)     upper = D * F   (call)
        lower = D * max(K - F, 0)   (put)      upper = D * K   (put)
    A quote outside (lower, upper) has no implied vol, so we report *why* rather
    than returning garbage. Such quotes are usually stale or mis-parsed data.
    """
    if not all(np.isfinite([price, F, K, T, D])) or min(F, K, T, D) <= 0 or price < 0:
        return IVResult(np.nan, "invalid_input")

    intrinsic = D * max(F - K, 0.0) if is_call else D * max(K - F, 0.0)
    upper = D * F if is_call else D * K
    # A small absolute cushion: prices within ~1e-12 of intrinsic carry no vol information.
    if price <= intrinsic + 1e-12 * max(1.0, F):
        return IVResult(np.nan, "below_intrinsic")
    if price >= upper:
        return IVResult(np.nan, "above_upper_bound")

    def objective(v: float) -> float:
        return float(black76_price(F, K, T, v, D, is_call)) - price

    f_lo, f_hi = objective(VOL_LOWER), objective(VOL_UPPER)
    if f_lo > 0 or f_hi < 0:  # price lies between the bounds but outside [1e-6, 500%] vol
        return IVResult(np.nan, "no_convergence")
    try:
        vol = brentq(objective, VOL_LOWER, VOL_UPPER, xtol=tol, maxiter=200)
    except (RuntimeError, ValueError):
        return IVResult(np.nan, "no_convergence")
    return IVResult(float(vol), "ok")


def implied_vol_array(
    price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike, D: ArrayLike, is_call: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised wrapper around :func:`implied_vol`; returns (vols, statuses)."""
    b = np.broadcast_arrays(*(np.asarray(x) for x in (price, F, K, T, D, is_call)))
    flat = [x.ravel() for x in b]
    results = [
        implied_vol(float(p), float(f), float(k), float(t), float(d), bool(c))
        for p, f, k, t, d, c in zip(*flat)
    ]
    vols = np.array([r.vol for r in results]).reshape(b[0].shape)
    statuses = np.array([r.status for r in results]).reshape(b[0].shape)
    return vols, statuses

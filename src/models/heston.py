"""Heston (1993) stochastic volatility model.

Risk-neutral dynamics (S = EUR/USD spot, v = instantaneous variance):

    dS_t / S_t = (r_d - r_f) dt + sqrt(v_t) dW^S_t
    dv_t       = kappa (theta - v_t) dt + sigma sqrt(v_t) dW^v_t,    corr(dW^S, dW^v) = rho

In words: variance is itself random. It is pulled back toward a long-run level ``theta``
at speed ``kappa`` (mean reversion), it is shaken by noise of size ``sigma`` ("vol of vol",
which fattens *both* tails -> smile curvature), and that noise is correlated with the
spot (``rho`` tilts the smile -> skew / risk reversal). ``v0`` is today's variance.

Characteristic function of X_T = ln(S_T / F) in the "Little Heston Trap" form of
Albrecher, Mayer, Schoutens & Tistaert (2007):

    beta = kappa - rho sigma i u,      d = sqrt(beta^2 + sigma^2 (u^2 + i u))
    g    = (beta - d) / (beta + d)
    A    = kappa theta / sigma^2 [ (beta - d) T - 2 ln( (1 - g e^{-dT}) / (1 - g) ) ]
    B    = (beta - d) / sigma^2 * (1 - e^{-dT}) / (1 - g e^{-dT})
    phi  = exp(A + B v0)

Heston's original paper writes the same function with ``g_orig = 1/g`` and ``e^{+dT}``.
Mathematically identical, but numerically the complex log in the original form jumps
across its branch cut for long maturities, silently producing wrong prices. In the
trap form |g e^{-dT}| < 1, so the log never wraps.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np


def _log1p_complex(z: np.ndarray) -> np.ndarray:
    """Accurate log(1 + z) for complex z.

    NumPy's complex ``log1p`` is computed as log(1 + z) and loses all precision for tiny z;
    here tiny z matters because we later divide by sigma^2. Taylor series for |z| < 1e-3
    (truncation error ~ |z|^7 relative to |z|).
    """
    z = np.asarray(z, dtype=complex)
    series = z * (1 - z * (1 / 2 - z * (1 / 3 - z * (1 / 4 - z * (1 / 5 - z / 6)))))
    with np.errstate(all="ignore"):
        direct = np.log(1 + z)
    return np.where(np.abs(z) < 1e-3, series, direct)


def heston_cf(
    u: np.ndarray, T: float, kappa: float, theta: float, sigma: float, rho: float, v0: float
) -> np.ndarray:
    """Heston CF of ln(S_T/F), stable formulation (u may be complex)."""
    u = np.asarray(u, dtype=complex)
    beta = kappa - rho * sigma * 1j * u
    d = np.sqrt(beta * beta + sigma * sigma * (u * u + 1j * u))
    # beta - d computed without cancellation: (beta^2 - d^2)/(beta + d) = -sigma^2 (u^2 + iu)/(beta + d).
    # This also cancels the 1/sigma^2 below, so the formula stays accurate as sigma -> 0.
    b_minus_d_over_s2 = -(u * u + 1j * u) / (beta + d)
    g = b_minus_d_over_s2 * sigma * sigma / (beta + d)  # = (beta - d)/(beta + d), cancellation-free
    edt = np.exp(-d * T)
    one_minus_edt = -np.expm1(-d * T)  # accurate for short maturities (dT small)
    B = b_minus_d_over_s2 * one_minus_edt / (1.0 - g * edt)
    # ln((1 - g e^{-dT}) / (1 - g)) = ln(1 + g (1 - e^{-dT}) / (1 - g)); g = O(sigma^2) when sigma is small.
    log_term = _log1p_complex(g * one_minus_edt / (1.0 - g))
    A = kappa * theta * (b_minus_d_over_s2 * T - 2.0 * log_term / (sigma * sigma))
    return np.exp(A + B * v0)


@dataclass(frozen=True)
class HestonParams:
    kappa: float  # mean-reversion speed of variance (per year); half-life = ln2 / kappa
    theta: float  # long-run variance (sqrt(theta) = long-run vol)
    sigma: float  # vol of vol
    rho: float    # correlation between spot and variance shocks
    v0: float     # current variance (sqrt(v0) = current instantaneous vol)

    NAMES = ("kappa", "theta", "sigma", "rho", "v0")

    def cf(self, u: np.ndarray, T: float) -> np.ndarray:
        return heston_cf(u, T, *astuple(self))

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / sigma^2. If > 1 the variance process can never hit zero (Feller condition).

        FX calibrations often *violate* this (ratio < 1): the market wants lots of vol-of-vol
        to produce the smile. That's allowed for pricing -- the CF is still valid -- but it
        makes Monte Carlo harder (variance hits 0 often), which is why we use full truncation.
        """
        return 2 * self.kappa * self.theta / self.sigma**2

    def expected_variance(self, T: float) -> float:
        """Risk-neutral expected average variance over [0, T]: the model's 'ATM-ish' vol^2 term structure."""
        if self.kappa * T < 1e-10:
            return self.v0
        return self.theta + (self.v0 - self.theta) * (1 - np.exp(-self.kappa * T)) / (self.kappa * T)

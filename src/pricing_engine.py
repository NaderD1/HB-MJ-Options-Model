"""Shared characteristic-function pricing engine.

Every model in this project is reduced to one object: the characteristic function (CF)
of the log-return relative to the forward,

    X_T = ln(S_T / F),        phi(u; T) = E[exp(i u X_T)]   (risk-neutral).

Because F is the risk-neutral mean of S_T, the CF must satisfy phi(-i; T) = E[S_T/F] = 1
(the "martingale condition"). If a model violates this, it prices the forward wrongly.

Pricing uses Lewis (2001)'s single-integral formula. Shifting the integration contour to
Im(u) = -1/2 makes the integrand decay like 1/u^2 and removes the singularity at u = 0:

    Call = D * [ F - sqrt(F K)/pi * Int_0^inf Re( exp(-i u k) phi(u - i/2) ) / (u^2 + 1/4) du ],
    k = ln(K / F).

Puts follow from put-call parity, C - P = D (F - K), which holds for *any* model.

A plug-in model only has to provide ``cf(u, T) -> complex array``.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import quad

CharFn = Callable[[np.ndarray, float], np.ndarray]


class Model(Protocol):
    """Anything with a ``cf(u, T)`` method is a model the engine can price."""

    def cf(self, u: np.ndarray, T: float) -> np.ndarray: ...


# Gauss–Legendre nodes on [-1, 1], reused for every panel.
_GL_X, _GL_W = np.polynomial.legendre.leggauss(16)
_U_SCAN = 2.0 ** np.arange(0, 18)  # 1 .. 131072


def _integration_limit(cf: CharFn, T: float, tol: float = 1e-13) -> float:
    """Smallest U on a doubling grid beyond which the Lewis integrand is negligible.

    The tail beyond U is bounded by  max|phi| * Int_U^inf du/u^2  ~  |phi(U)| / U.
    Short maturities need a large U (the density is narrow, so its CF is wide):
    a 1-week option at 8% vol needs U ~ 500; a 10-year option needs U ~ 30.
    """
    mag = np.abs(cf(_U_SCAN - 0.5j, T))
    ok = mag / _U_SCAN < tol
    # Require the condition to hold from that point on, not just at one lucky node.
    tail_ok = np.flip(np.logical_and.accumulate(np.flip(ok)))
    idx = np.argmax(tail_ok) if tail_ok.any() else len(_U_SCAN) - 1
    return float(max(_U_SCAN[idx], 50.0))


def _lewis_nodes(U: float, k_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss–Legendre nodes/weights on [0, U].

    The 1/(u^2 + 1/4) factor has complex poles at u = +-i/2, right next to the start of
    the interval, so panels are graded finely near 0. Beyond that, panel width is capped
    so that exp(-i u k) is resolved for the widest strike.
    """
    width = min(5.0, np.pi / max(k_max, 1e-8))
    near = np.array([0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    near = near[near < U]
    n_far = int(min(np.ceil((U - near[-1]) / width), 4000))
    edges = np.concatenate([near, np.linspace(near[-1], U, n_far + 1)[1:]])
    a, b = edges[:-1, None], edges[1:, None]
    u = (0.5 * (b - a) * _GL_X + 0.5 * (a + b)).ravel()
    w = (0.5 * (b - a) * _GL_W).ravel()
    return u, w


def call_prices_single_T(cf: CharFn, F: float, K: np.ndarray, T: float, D: float) -> np.ndarray:
    """Vectorised Lewis call prices for many strikes at one maturity."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    k = np.log(K / F)
    u, w = _lewis_nodes(_integration_limit(cf, T), float(np.max(np.abs(k))))
    phi = cf(u - 0.5j, T)
    integrand = np.real(np.exp(-1j * np.outer(k, u)) * phi) / (u**2 + 0.25)
    integral = integrand @ w
    return D * (F - np.sqrt(F * K) / np.pi * integral)


def price_european(
    model: Model | CharFn,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    D: ArrayLike,
    is_call: ArrayLike,
) -> np.ndarray:
    """Price European options under any CF model (vectorised; groups by maturity)."""
    cf: CharFn = model.cf if hasattr(model, "cf") else model  # type: ignore[union-attr]
    F, K, T, D, is_call = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (F, K, T, D)), np.asarray(is_call, dtype=bool)
    )
    out = np.empty(K.shape, dtype=float)
    # Options sharing (T, F, D) share one CF evaluation.
    keys = np.stack([T.ravel(), F.ravel(), D.ravel()], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    flat_out = out.ravel()
    for j, (t, f, d) in enumerate(uniq):
        sel = inv.ravel() == j
        flat_out[sel] = call_prices_single_T(cf, f, K.ravel()[sel], t, d)
    calls = flat_out.reshape(K.shape)
    puts = calls - D * (F - K)
    return np.where(is_call, calls, puts)


def price_european_quad(cf: CharFn, F: float, K: float, T: float, D: float, is_call: bool) -> float:
    """Slow reference pricer: Gil-Pelaez two-probability form with adaptive quadrature.

    A *different* formula from the Lewis one above, used only in tests so the two
    implementations check each other:

        Call = D [F P1 - K P2]
        P2 = 1/2 + 1/pi Int Re[ exp(-i u k) phi(u)     / (i u) ] du   (= Q(S_T > K))
        P1 = 1/2 + 1/pi Int Re[ exp(-i u k) phi(u - i) / (i u) ] du   (share-measure prob.)
    """
    k = np.log(K / F)

    def p(shift: complex) -> float:
        f = lambda u: np.real(np.exp(-1j * u * k) * cf(np.array([u + shift]), T)[0] / (1j * u))
        return 0.5 + quad(f, 1e-12, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-11)[0] / np.pi

    call = D * (F * p(-1j) - K * p(0.0))
    return float(call if is_call else call - D * (F - K))


def black_scholes_cf(vol: float) -> CharFn:
    """CF of X_T under constant vol: X_T ~ N(-vol^2 T/2, vol^2 T).

    phi(u) = exp(-1/2 vol^2 T (u^2 + i u)).  Used to test the engine against closed form.
    """
    return lambda u, T: np.exp(-0.5 * vol**2 * T * (u * u + 1j * u))

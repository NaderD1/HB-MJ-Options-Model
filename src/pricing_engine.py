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
from scipy.interpolate import CubicSpline

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
    # Width grows with the integration range (>= 100 panels over [0, U]); short expiries have U ~ 4096,
    # so this cuts nodes ~8x vs a fixed width of 5. U/40 was too coarse: jump factors add features on
    # scales 1/sigma_J and 2 pi/|mu_J| (~70-300) that 100-wide panels miss -- see test_bates/test_hbmj.
    width = min(max(5.0, U / 100.0), np.pi / max(k_max, 1e-8))
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


# --------------------------------------------------------------------------- #
# Carr–Madan (1999) FFT
# --------------------------------------------------------------------------- #

def carr_madan_grid(
    cf: CharFn, T: float, alpha: float = 0.75, eta_max: float = 0.1, n_max: int = 2**21
) -> tuple[np.ndarray, np.ndarray]:
    """Normalised call prices c(k) = E[(e^X - e^k)^+] on an FFT log-strike grid.

    Idea: the call price C(k) is not integrable in k (it tends to F as k -> -inf), so it has
    no Fourier transform. Carr & Madan multiply by a damping factor e^{alpha k}; the damped
    price does have one, and it is known in closed form from the CF:

        psi(v) = phi(v - (alpha+1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v)
        c(k)   = e^{-alpha k} / pi * Int_0^inf Re[ e^{-i v k} psi(v) ] dv

    Sampling v_j = eta*j (j < N) and k_m = -b + lambda*m with lambda*eta = 2 pi / N turns the
    integral into ONE discrete Fourier transform, giving c(k) at all N strikes at once.

    Grid choice. Because lambda = 2 pi / (N eta) = 2 pi / v_max, a fine strike grid needs a
    long frequency range. Short-dated FX smiles span only ~1% in log-strike, so lambda must be
    small relative to the distribution width; we tie it to the integration limit U (which
    scales like 1/width). Simpson weights give O(eta^4) quadrature error.
    """
    # alpha is only admissible if E[(S_T/F)^{alpha+1}] is finite (Lee 2004). Past a moment
    # explosion the closed-form CF does not return inf -- it returns a finite *complex* number,
    # which cannot be the moment of a positive variable. So: must be finite, real and positive.
    m = cf(np.array([-(alpha + 1.0) * 1j]), T)[0]
    if not np.isfinite(m) or np.real(m) <= 0 or abs(np.imag(m)) > 1e-8 * abs(np.real(m)):
        raise ValueError(f"Moment E[(S_T/F)^{alpha + 1}] looks infinite at T={T}; lower alpha.")

    U = _integration_limit(cf, T)
    lam = min(0.005, 0.4 / U)                  # log-strike spacing: >= ~15 points per std dev
    v_max = max(2 * np.pi / lam, U)
    N = int(2 ** np.ceil(np.log2(v_max / eta_max)))
    if N > n_max:
        # Never degrade silently: capping N coarsens eta and biases every price (seen with high
        # vol-of-vol at 1W: a constant -2.6e-5 offset). The FFT is a validation tool -- fail loudly.
        raise ValueError(f"Carr-Madan grid needs N={N} > n_max={n_max} at T={T}; raise n_max or use integration")
    eta = v_max / N
    lam = 2 * np.pi / (N * eta)
    b = N * lam / 2

    v = eta * np.arange(N)
    psi = cf(v - (alpha + 1) * 1j, T) / (alpha**2 + alpha - v**2 + 1j * (2 * alpha + 1) * v)
    simpson = (3 + (-1.0) ** (np.arange(N) + 1)) / 3
    simpson[0] = 1 / 3
    x = np.exp(1j * b * v) * psi * eta * simpson
    k = -b + lam * np.arange(N)
    c = np.exp(-alpha * k) / np.pi * np.real(np.fft.fft(x))
    return k, c


def call_prices_single_T_fft(
    cf: CharFn, F: float, K: np.ndarray, T: float, D: float, alpha: float = 0.75
) -> np.ndarray:
    """Carr–Madan FFT call prices, cubic-spline interpolated from the grid to the requested strikes."""
    K = np.atleast_1d(np.asarray(K, dtype=float))
    k_target = np.log(K / F)
    k, c = carr_madan_grid(cf, T, alpha)
    lo = max(np.searchsorted(k, k_target.min()) - 8, 0)
    hi = min(np.searchsorted(k, k_target.max()) + 8, len(k))
    if k_target.min() < k[0] or k_target.max() > k[-1]:
        raise ValueError("Requested strikes fall outside the FFT log-strike grid.")
    return D * F * CubicSpline(k[lo:hi], c[lo:hi])(k_target)


def price_european(
    model: Model | CharFn,
    F: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    D: ArrayLike,
    is_call: ArrayLike,
    method: str = "integration",
) -> np.ndarray:
    """Price European options under any CF model (vectorised; groups by maturity).

    method: "integration" (Lewis single integral, default) or "fft" (Carr–Madan).
    """
    cf: CharFn = model.cf if hasattr(model, "cf") else model  # type: ignore[union-attr]
    pricer = {"integration": call_prices_single_T, "fft": call_prices_single_T_fft}[method]
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
        flat_out[sel] = pricer(cf, f, K.ravel()[sel], t, d)
    calls = flat_out.reshape(K.shape)
    puts = calls - D * (F - K)
    return np.where(is_call, calls, puts)


def price_european_quad(model: Model | CharFn, F: float, K: float, T: float, D: float, is_call: bool) -> float:
    """Slow reference pricer: Gil-Pelaez two-probability form with adaptive quadrature.

    A *different* formula from the Lewis one above, used only in tests so the two
    implementations check each other:

        Call = D [F P1 - K P2]
        P2 = 1/2 + 1/pi Int Re[ exp(-i u k) phi(u)     / (i u) ] du   (= Q(S_T > K))
        P1 = 1/2 + 1/pi Int Re[ exp(-i u k) phi(u - i) / (i u) ] du   (share-measure prob.)
    """
    cf: CharFn = model.cf if hasattr(model, "cf") else model  # type: ignore[union-attr]
    k = np.log(K / F)

    def p(shift: complex) -> float:
        f = lambda u: np.real(np.exp(-1j * u * k) * cf(np.array([u + shift]), T)[0] / (1j * u))
        return 0.5 + quad(f, 1e-12, np.inf, limit=2000, epsabs=1e-13, epsrel=1e-11)[0] / np.pi

    call = D * (F * p(-1j) - K * p(0.0))
    return float(call if is_call else call - D * (F - K))

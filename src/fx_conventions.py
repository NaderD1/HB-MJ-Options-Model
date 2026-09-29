"""FX option market conventions: deltas, ATM definitions, and smile reconstruction.

References: Clark (2011), *Foreign Exchange Option Pricing*, ch. 3; Reiswich & Wystup (2010),
"A guide to FX options quoting conventions", J. Derivatives 18(2).

Notation: pair FOR/DOM = EUR/USD (1 EUR costs S USD). F forward, D_d = USD discount factor,
D_f = EUR discount factor, omega = +1 call (on EUR), -1 put.
    d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt T),  d2 = d1 - sigma sqrt T

Delta conventions (all four exist in the market; which one applies is a *data* fact):
    spot            omega * D_f * N(omega d1)          premium in DOM (USD), <= 1Y for EUR/USD
    forward         omega * N(omega d1)                premium in DOM, long tenors
    spot_pa         omega * D_f * (K/F) * N(omega d2)  premium in FOR (EUR), e.g. USD/JPY
    forward_pa      omega * (K/F) * N(omega d2)
"PA" = premium-adjusted: when the premium is paid in the foreign currency, the delta hedge
must also cover the premium. Standard EUR/USD: premium in USD -> *unadjusted* delta.

ATM conventions:
    atmf  K = F
    dns   delta-neutral straddle: call delta + put delta = 0
          unadjusted -> K = F exp(+sigma^2 T/2);  premium-adjusted -> K = F exp(-sigma^2 T/2)
    spot  K = S (rare)

Quotes -> smile (RR = sigma_call - sigma_put at the same |delta|, EUR-call-over convention):
    "smile" butterfly:  sigma_25C = ATM + BF + RR/2,  sigma_25P = ATM + BF - RR/2
    "market"/"broker" butterfly (a.k.a. market strangle): BF_MS is the vol spread at which a
          25D strangle, struck with ONE vol (ATM + BF_MS) at that vol's own 25D strikes, is
          priced. Converting to the smile requires an interpolated smile; the answer depends
          (mildly) on the interpolation. We use a natural cubic spline in ln(K/F) through
          the pillars, flat beyond them, and document that choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, root
from scipy.stats import norm

from src.models.black_scholes import black76_price

DeltaType = Literal["spot", "forward", "spot_pa", "forward_pa"]
AtmType = Literal["atmf", "dns", "spot"]
BfType = Literal["smile", "market"]
DELTA_TYPES = ("spot", "forward", "spot_pa", "forward_pa")
ATM_TYPES = ("atmf", "dns", "spot")
BF_TYPES = ("smile", "market")


@dataclass(frozen=True)
class FXConventions:
    delta_type: DeltaType
    atm_type: AtmType
    bf_type: BfType
    premium_currency: str  # "USD" (domestic) or "EUR" (foreign) for EUR/USD

    def __post_init__(self) -> None:
        if self.delta_type not in DELTA_TYPES or self.atm_type not in ATM_TYPES or self.bf_type not in BF_TYPES:
            raise ValueError(f"Unknown convention in {self}")

    @property
    def premium_adjusted(self) -> bool:
        return self.delta_type.endswith("_pa")


def _d1d2(F, K, T, vol):
    sd = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return d1, d1 - sd


def fx_delta(F: float, K: float, T: float, vol: float, D_f: float, omega: int, delta_type: DeltaType) -> float:
    d1, d2 = _d1d2(F, K, T, vol)
    df = D_f if delta_type.startswith("spot") else 1.0
    if delta_type.endswith("_pa"):
        return float(omega * df * (K / F) * norm.cdf(omega * d2))
    return float(omega * df * norm.cdf(omega * d1))


def strike_from_delta(delta: float, F: float, T: float, vol: float, D_f: float, omega: int, delta_type: DeltaType) -> float:
    """Invert the delta convention for the strike (delta is signed: puts negative)."""
    sd = vol * np.sqrt(T)
    df = D_f if delta_type.startswith("spot") else 1.0
    if not delta_type.endswith("_pa"):
        return float(F * np.exp(-omega * sd * norm.ppf(omega * delta / df) + 0.5 * sd * sd))
    f = lambda K: fx_delta(F, K, T, vol, D_f, omega, delta_type) - delta
    k_unadj = F * np.exp(-omega * sd * norm.ppf(omega * delta / df) + 0.5 * sd * sd)
    if omega < 0:  # put PA delta is monotone in K
        return float(brentq(f, F * np.exp(-12 * sd), k_unadj * np.exp(12 * sd), xtol=1e-14 * F))
    # Call PA delta rises then falls in K; the quoted strike is on the right branch, between the
    # delta maximum K_min (sd N(d2) = n(d2)) and the unadjusted strike K_max (Reiswich-Wystup).
    g = lambda K: sd * norm.cdf(_d1d2(F, K, T, vol)[1]) - norm.pdf(_d1d2(F, K, T, vol)[1])
    k_min = brentq(g, F * np.exp(-12 * sd), k_unadj, xtol=1e-14 * F)
    return float(brentq(f, k_min, k_unadj, xtol=1e-14 * F))


def atm_strike(F: float, S: float, T: float, vol: float, atm_type: AtmType, premium_adjusted: bool) -> float:
    if atm_type == "atmf":
        return F
    if atm_type == "spot":
        return S
    return float(F * np.exp((-0.5 if premium_adjusted else 0.5) * vol * vol * T))


# --------------------------------------------------------------------------- #
# Smile reconstruction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SmilePoint:
    bucket: str       # "ATM", "25DC", "25DP", "10DC", "10DP"
    delta: float      # signed quoted delta (ATM: 0 by construction for DNS)
    is_call: bool
    vol: float
    strike: float


def _pillar_strikes(atm, wings, F, S, T, D_f, conv):
    """Strikes for ATM and each (delta, sigma_call, sigma_put) wing, each at its own vol."""
    pts = [SmilePoint("ATM", 0.0, True, atm, atm_strike(F, S, T, atm, conv.atm_type, conv.premium_adjusted))]
    for d, s_c, s_p in wings:
        tag = f"{round(100 * d)}D"
        pts.append(SmilePoint(tag + "P", -d, False, s_p, strike_from_delta(-d, F, T, s_p, D_f, -1, conv.delta_type)))
        pts.append(SmilePoint(tag + "C", d, True, s_c, strike_from_delta(d, F, T, s_c, D_f, 1, conv.delta_type)))
    return pts


def smile_interpolator(points: list[SmilePoint], F: float):
    """sigma(K): natural cubic spline in ln(K/F) through the pillars, flat outside."""
    x = np.array([np.log(p.strike / F) for p in points])
    v = np.array([p.vol for p in points])
    order = np.argsort(x)
    x, v = x[order], v[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Smile pillar strikes are not strictly increasing")
    spline = CubicSpline(x, v, bc_type="natural")
    return lambda K: spline(np.clip(np.log(np.asarray(K) / F), x[0], x[-1]))


def _market_strangle_value(delta, vol_ms, F, S, T, D_d, D_f, conv):
    """Strikes and premium of the market strangle: both legs at vol_ms, at vol_ms's own delta strikes."""
    kc = strike_from_delta(delta, F, T, vol_ms, D_f, 1, conv.delta_type)
    kp = strike_from_delta(-delta, F, T, vol_ms, D_f, -1, conv.delta_type)
    v = black76_price(F, kc, T, vol_ms, D_d, True) + black76_price(F, kp, T, vol_ms, D_d, False)
    return kc, kp, float(v)


def build_smile(
    atm: float,
    quotes: dict[float, tuple[float, float]],   # delta -> (RR, BF), e.g. {0.25: (rr25, bf25), 0.10: (rr10, bf10)}
    F: float, S: float, T: float, D_d: float, D_f: float, conv: FXConventions,
) -> list[SmilePoint]:
    """Turn ATM/RR/BF vol quotes into strike-vol points under the stated conventions."""
    deltas = sorted(quotes, reverse=True)

    def wings(bf_smile):
        return [(d, atm + bf_smile[i] + quotes[d][0] / 2, atm + bf_smile[i] - quotes[d][0] / 2) for i, d in enumerate(deltas)]

    if conv.bf_type == "smile":
        return _pillar_strikes(atm, wings([quotes[d][1] for d in deltas]), F, S, T, D_f, conv)

    # Market strangle: find smile-strangle spreads so that the interpolated smile reprices each
    # market strangle at its own (single-vol) strikes.
    targets = [_market_strangle_value(d, atm + quotes[d][1], F, S, T, D_d, D_f, conv) for d in deltas]

    def residual(bf_smile):
        pts = _pillar_strikes(atm, wings(bf_smile), F, S, T, D_f, conv)
        sig = smile_interpolator(pts, F)
        out = []
        for kc, kp, v in targets:
            model_v = black76_price(F, kc, T, float(sig(kc)), D_d, True) + black76_price(F, kp, T, float(sig(kp)), D_d, False)
            out.append((model_v - v) / (D_d * F))  # scale: fraction of forward
        return np.array(out)

    sol = root(residual, x0=[quotes[d][1] for d in deltas], method="hybr", tol=1e-14)
    # Judge convergence by the residual itself (hybr can report "slow progress" at 1e-15 levels).
    if np.max(np.abs(residual(sol.x))) > 1e-11:
        raise RuntimeError(f"Market-strangle conversion failed: {sol.message}")
    return _pillar_strikes(atm, wings(sol.x), F, S, T, D_f, conv)


def quotes_from_smile(
    points: list[SmilePoint], F: float, S: float, T: float, D_d: float, D_f: float, conv: FXConventions
) -> dict:
    """Inverse of build_smile: recover ATM, RR and BF (in the data's BF convention) from points.

    Used to verify that normalization reproduces the original Bloomberg quotes.
    """
    by = {p.bucket: p for p in points}
    out = {"atm": by["ATM"].vol}
    sig = smile_interpolator(points, F)
    for tag, d in (("25", 0.25), ("10", 0.10)):
        if f"{tag}DC" not in by:
            continue
        c, p = by[f"{tag}DC"].vol, by[f"{tag}DP"].vol
        out[f"rr{tag}"] = c - p
        if conv.bf_type == "smile":
            out[f"bf{tag}"] = 0.5 * (c + p) - out["atm"]
        else:
            def gap(bf_ms):
                kc, kp, v = _market_strangle_value(d, out["atm"] + bf_ms, F, S, T, D_d, D_f, conv)
                return black76_price(F, kc, T, float(sig(kc)), D_d, True) + black76_price(F, kp, T, float(sig(kp)), D_d, False) - v
            guess = 0.5 * (c + p) - out["atm"]
            out[f"bf{tag}"] = brentq(gap, guess - 0.05, guess + 0.05, xtol=1e-15)
    return out

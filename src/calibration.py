"""Calibration engine for the nested models: Heston -> Bates -> HB-MJ.

Design
------
* One engine for all three models. A ``ModelSpec`` lists the free parameters and builds a
  :class:`~src.models.nested.NestedModel` from a parameter dict; everything else is shared.
* Production pricer: the direct (Lewis) integration engine. FFT and Monte Carlo are used only
  to verify a final fit on a subset of contracts (``verify_fit``).
* Optimiser: ``scipy.optimize.least_squares`` (trust-region reflective, box bounds) on a
  vector of weighted residuals, run from several deliberately different starting points.
  Least squares gives a Jacobian at the solution, which the identifiability diagnostics use.
* Parameters are optimised in a transformed space z:
      log  for strictly positive scale parameters (v0, kappa, theta, sigma)
      atanh for rho in (-1, 1)
      linear/scaled for parameters that must be able to reach exactly 0 (lam, sigma_J, sigma_E)
      and for mu_J (bounded, either sign)
  The transformed box is handed to the optimiser, so invalid regions are never evaluated.
* The Feller condition is NOT imposed; its ratio is recorded for every fit.

Objective (see ``Objective``)
-----------------------------
  kind="iv"    residual_i = sqrt(w_i) * (IV_model_i - IV_market_i)          (primary)
  kind="price" residual_i = sqrt(w_i) * (P_model_i - P_market_i) / (D_i F_i) (diagnostic)
Weights w_i (normalised to mean 1):
  "vega"            vega_i / (largest vega in the same expiry). Deep wings (low vega, noisy IV)
                    get less weight, but short expiries are NOT down-weighted relative to long
                    ones -- raw vega grows like sqrt(T) and would bury the 1W event contracts.
                    DEFAULT.
  "bidask"          1 / max(ask - bid, floor)
  "inv_bidask_var"  1 / max(ask - bid, floor)^2
  "equal"           1
Invalid model prices (NaN, outside no-arbitrage bounds, IV failure) get a large fixed residual
so the optimiser is pushed away instead of crashing.

Hierarchy (``calibrate_nested``)
--------------------------------
  1. Heston on all contracts.
  2. Bates, started from the Heston fit plus several jump starts (including lam ~ 0).
  3. HB-MJ, started from the Bates fit:
       3a. event-only stage: Heston/Bates parameters fixed at the Bates fit, only the event
           sigmas move (the incremental effect of the event component);
       3b. joint refinement of everything.
     An event type is a free parameter only if at least one contract spans an event of that
     type; otherwise it is fixed at 0 and reported as not identifiable. Because the event
     factor is exactly 1 for contracts that do not span an event, non-event contracts inform
     only the Heston/Bates parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc

from src.events import EVENT_TYPES, ScheduledEvent, select_events_before_expiry
from src.metrics import ContractSet
from src.models.bates import PoissonJumps
from src.models.black_scholes import black76_price, black76_vega, implied_vol_fast
from src.models.hbmj import ScheduledEventJumps
from src.models.heston import HestonParams
from src.models.nested import NestedModel
from src.pricing_engine import price_european

INVALID_RESIDUAL = 1.0  # 100 vol points: far larger than any sensible fit error
# A parameter within 0.1% of its (transformed) bound range counts as a bound hit. (1e-6 was too tight:
# it missed lambda = 49.9999 of 50 and theta = 0.24996 of 0.25 in the proxy study.)
BOUND_TOL = 1e-3


# =========================================================================== #
# Parameters and transforms
# =========================================================================== #

@dataclass(frozen=True)
class ParamSpec:
    name: str
    lower: float
    upper: float
    transform: str        # "log" | "atanh" | "linear"
    scale: float = 1.0    # for "linear": z = x / scale (puts all parameters on O(1) scales)

    def to_z(self, x: float) -> float:
        if self.transform == "log":
            return float(np.log(x))
        if self.transform == "atanh":
            return float(np.arctanh(x))
        return float(x / self.scale)

    def from_z(self, z: float) -> float:
        if self.transform == "log":
            return float(np.exp(z))
        if self.transform == "atanh":
            return float(np.tanh(z))
        return float(z * self.scale)

    def z_bounds(self) -> tuple[float, float]:
        return self.to_z(self.lower), self.to_z(self.upper)

    def clip(self, x: float) -> float:
        """Move a value strictly inside the bounds (least_squares needs a strictly feasible start)."""
        lo, hi = self.z_bounds()
        eps = 1e-6 * (hi - lo)
        z = self.to_z(float(np.clip(x, self.lower, self.upper)))
        return self.from_z(float(np.clip(z, lo + eps, hi - eps)))

    def typical(self, x: float) -> float:
        """Typical magnitude used to scale Jacobian columns."""
        if self.transform == "linear":
            return max(abs(x), self.scale)
        if self.transform == "atanh":
            return max(abs(x), 0.1)
        return abs(x)


HESTON_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("v0", 1e-5, 0.25, "log"),        # instantaneous vol 0.3% .. 50%
    ParamSpec("kappa", 0.01, 30.0, "log"),     # half-life 0.02y .. 70y
    ParamSpec("theta", 1e-5, 0.25, "log"),
    ParamSpec("sigma", 0.01, 5.0, "log"),      # vol of vol
    ParamSpec("rho", -0.99, 0.99, "atanh"),
)
JUMP_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("lam", 0.0, 50.0, "linear", 1.0),       # jumps per year; 0 allowed (collapses to Heston)
    ParamSpec("mu_J", -0.10, 0.10, "linear", 0.01),   # mean log jump +-10%
    ParamSpec("sigma_J", 0.0, 0.10, "linear", 0.01),  # jump size dispersion
)
EVENT_PARAMS: tuple[ParamSpec, ...] = tuple(
    ParamSpec(f"sigma_{k}", 0.0, 0.03, "linear", 0.002) for k in EVENT_TYPES  # up to a 3% event move (1 sd)
)
SPECS_BY_NAME = {p.name: p for p in HESTON_PARAMS + JUMP_PARAMS + EVENT_PARAMS}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: tuple[ParamSpec, ...]

    def build(self, x: Mapping[str, float], events: tuple[ScheduledEvent, ...]) -> NestedModel:
        h = HestonParams(x["kappa"], x["theta"], x["sigma"], x["rho"], x["v0"])
        jumps = PoissonJumps(x["lam"], x["mu_J"], x["sigma_J"]) if "lam" in x else None
        ev = None
        if any(k.startswith("sigma_") and k != "sigma_J" for k in x):
            ev = ScheduledEventJumps(events, {k: x.get(f"sigma_{k}", 0.0) for k in EVENT_TYPES})
        return NestedModel(h, jumps, ev)


HESTON = ModelSpec("Heston", HESTON_PARAMS)
BATES = ModelSpec("Bates", HESTON_PARAMS + JUMP_PARAMS)
HBMJ = ModelSpec("HB-MJ", HESTON_PARAMS + JUMP_PARAMS + EVENT_PARAMS)


# =========================================================================== #
# Data
# =========================================================================== #

@dataclass(frozen=True)
class CalibrationData:
    """Market (or synthetic) quotes on fixed contracts, plus the event schedule for the valuation."""

    contracts: ContractSet
    iv: np.ndarray                      # market mid implied vol
    price: np.ndarray                   # market mid price (Black-76 of iv)
    events: tuple[ScheduledEvent, ...]  # as-known schedule at valuation (tau in years)
    iv_bid: np.ndarray | None = None
    iv_ask: np.ndarray | None = None
    labels: tuple[str, ...] = ()        # e.g. tenor per contract (for reports)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_spans", tuple(
            select_events_before_expiry(self.events, float(t)) for t in self.contracts.T
        ))

    @property
    def spanned_events(self) -> tuple[tuple[ScheduledEvent, ...], ...]:
        """Per contract: the events it spans -- via the same rule as pricing and data tagging."""
        return self._spans  # type: ignore[attr-defined]

    def spans_type(self, kind: str) -> np.ndarray:
        return np.array([any(e.kind == kind for e in s) for s in self.spanned_events])

    @property
    def event_spanning(self) -> np.ndarray:
        return np.array([len(s) > 0 for s in self.spanned_events])

    def identifiable_event_types(self) -> tuple[str, ...]:
        return tuple(k for k in EVENT_TYPES if self.spans_type(k).any())

    def subset(self, mask: np.ndarray) -> "CalibrationData":
        c = self.contracts
        cs = ContractSet.from_arrays(c.F[mask], c.K[mask], c.T[mask], c.D[mask], c.is_call[mask])
        pick = lambda a: None if a is None else np.asarray(a)[mask]
        return CalibrationData(cs, self.iv[mask], self.price[mask], self.events, pick(self.iv_bid), pick(self.iv_ask),
                               tuple(np.asarray(self.labels)[mask]) if self.labels else ())


def synthetic_surface(
    model: NestedModel, expiries: Sequence[float], events: tuple[ScheduledEvent, ...] = (),
    z_grid: Sequence[float] = (-1.8, -0.95, 0.0, 0.95, 1.8), ref_vol: float = 0.08,
    spot: float = 1.16, r_usd: float = 0.038, r_eur: float = 0.021, half_spread_vol: float = 0.002,
    noise_vol: float = 0.0, seed: int = 0, labels: Sequence[str] | None = None,
) -> CalibrationData:
    """Contracts at fixed log-moneyness k = z * ref_vol * sqrt(T) (roughly 10D/25D/ATM/25D/10D),
    priced by a known model. Optional seeded Gaussian IV noise. The event schedule stored in the
    data must be the same one the true model prices with (checked)."""
    if model.events is not None and tuple(model.events.events) != tuple(events):
        raise ValueError("synthetic_surface: model events and data events differ")
    T = np.repeat(np.asarray(expiries, float), len(z_grid))
    z = np.tile(np.asarray(z_grid, float), len(expiries))
    F = spot * np.exp((r_usd - r_eur) * T)
    D = np.exp(-r_usd * T)
    K = F * np.exp(z * ref_vol * np.sqrt(T))
    is_call = K >= F
    price = price_european(model, F, K, T, D, is_call)
    iv, ok = implied_vol_fast(price, F, K, T, D, is_call, guess=np.full(T.shape, ref_vol))
    if not ok.all():
        raise ValueError("synthetic_surface: IV inversion failed for some contracts")
    if noise_vol > 0:
        iv = iv + np.random.default_rng(seed).normal(0.0, noise_vol, iv.shape)
        price = black76_price(F, K, T, iv, D, is_call)
    lab = tuple(np.repeat(np.asarray(labels), len(z_grid))) if labels is not None else ()
    return CalibrationData(ContractSet.from_arrays(F, K, T, D, is_call), iv, price, events,
                           iv - half_spread_vol, iv + half_spread_vol, lab)


# =========================================================================== #
# Objective
# =========================================================================== #

@dataclass(frozen=True)
class Objective:
    kind: str = "iv"            # "iv" (primary) | "price" (diagnostic)
    weighting: str = "vega"     # "vega" | "bidask" | "inv_bidask_var" | "equal"
    spread_floor: float = 0.001  # 0.1 vol pt: stops a near-zero quoted spread from exploding a weight


def weights(data: CalibrationData, obj: Objective) -> np.ndarray:
    c = data.contracts
    if obj.weighting == "equal":
        w = np.ones(len(c))
    elif obj.weighting == "vega":
        vega = black76_vega(c.F, c.K, c.T, data.iv, c.D)
        w = np.empty(len(c))
        for t in np.unique(c.T):
            m = c.T == t
            w[m] = vega[m] / vega[m].max()
    elif obj.weighting in ("bidask", "inv_bidask_var"):
        if data.iv_bid is None or data.iv_ask is None:
            raise ValueError(f"weighting={obj.weighting!r} needs bid/ask vols")
        s = np.maximum(np.asarray(data.iv_ask) - np.asarray(data.iv_bid), obj.spread_floor)
        w = 1.0 / s if obj.weighting == "bidask" else 1.0 / s**2
    else:
        raise ValueError(f"unknown weighting {obj.weighting!r}")
    return w / w.mean()


@dataclass
class Evaluation:
    residuals: np.ndarray
    iv_model: np.ndarray
    price_model: np.ndarray
    n_invalid: int


def evaluate(model: NestedModel, data: CalibrationData, obj: Objective, w: np.ndarray | None = None) -> Evaluation:
    c = data.contracts
    w = weights(data, obj) if w is None else w
    price = price_european(model, c.F, c.K, c.T, c.D, c.is_call)
    iv, ok = implied_vol_fast(price, c.F, c.K, c.T, c.D, c.is_call, guess=data.iv)
    ok &= np.isfinite(price)
    if obj.kind == "iv":
        raw = np.where(ok, iv - data.iv, INVALID_RESIDUAL)
    elif obj.kind == "price":
        raw = np.where(np.isfinite(price), (price - data.price) / (c.D * c.F), INVALID_RESIDUAL)
    else:
        raise ValueError(f"unknown objective kind {obj.kind!r}")
    return Evaluation(np.sqrt(w) * raw, iv, price, int((~ok).sum()))


# =========================================================================== #
# Fitting
# =========================================================================== #

@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    free: tuple[str, ...]
    fixed: dict[str, float]
    start: dict[str, float]
    cost: float                 # 0.5 * sum(weighted residual^2)
    iv_rmse: float              # unweighted, decimal vol
    n_eval: int
    status: int
    message: str
    success: bool
    seconds: float
    bound_hits: tuple[str, ...]
    feller_ratio: float
    n_invalid: int
    jac_z: np.ndarray = field(repr=False, default=None)  # Jacobian w.r.t. transformed free params

    def summary(self) -> dict:
        return {"model": self.model, **{k: self.params[k] for k in self.params}, "cost": self.cost,
                "iv_rmse_bp": 1e4 * self.iv_rmse, "n_eval": self.n_eval, "status": self.status,
                "success": self.success, "bound_hits": ";".join(self.bound_hits), "feller": self.feller_ratio,
                "n_invalid": self.n_invalid, "seconds": self.seconds}


def fit(
    spec: ModelSpec, data: CalibrationData, obj: Objective, start: Mapping[str, float],
    fixed: Mapping[str, float] | None = None, max_nfev: int = 400,
) -> FitResult:
    """One least-squares fit from one start. `fixed` parameters are held constant."""
    fixed = dict(fixed or {})
    free = tuple(p.name for p in spec.params if p.name not in fixed)
    fspecs = [SPECS_BY_NAME[n] for n in free]
    w = weights(data, obj)
    x0 = {n: SPECS_BY_NAME[n].clip(float(start[n])) for n in free}
    z0 = np.array([s.to_z(x0[s.name]) for s in fspecs])
    lb, ub = np.array([s.z_bounds() for s in fspecs]).T

    def unpack(z):
        return {**fixed, **{s.name: s.from_z(zi) for s, zi in zip(fspecs, z)}}

    def fun(z):
        return evaluate(spec.build(unpack(z), data.events), data, obj, w).residuals

    t0 = time.perf_counter()
    res = least_squares(fun, z0, bounds=(lb, ub), method="trf", x_scale="jac", max_nfev=max_nfev,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    x = unpack(res.x)
    ev = evaluate(spec.build(x, data.events), data, obj, w)
    hits = tuple(s.name for s, zi, lo, hi in zip(fspecs, res.x, lb, ub)
                 if min(zi - lo, hi - zi) < BOUND_TOL * (hi - lo))
    params = {p.name: x[p.name] for p in spec.params}
    iv_err = ev.iv_model - data.iv
    return FitResult(
        spec.name, params, free, fixed, {**fixed, **x0}, float(0.5 * np.sum(ev.residuals**2)),
        float(np.sqrt(np.nanmean(iv_err**2))), int(res.nfev), int(res.status), str(res.message), bool(res.success),
        time.perf_counter() - t0, hits, 2 * params["kappa"] * params["theta"] / params["sigma"] ** 2,
        ev.n_invalid, res.jac,
    )


def random_starts(spec: ModelSpec, n: int, seed: int, base: Mapping[str, float] | None = None,
                  names: Iterable[str] | None = None) -> list[dict[str, float]]:
    """n Sobol-spread starts in the transformed box (fixed seed). With `base`, only `names` vary."""
    names = list(names) if names is not None else [p.name for p in spec.params]
    specs = [SPECS_BY_NAME[k] for k in names]
    m = int(np.ceil(np.log2(max(n, 1))))  # Sobol balance needs a power of 2; draw 2^m, keep n
    u = qmc.Sobol(len(specs), scramble=True, seed=seed).random_base2(m)[:n]
    out = []
    for row in u:
        s = dict(base or {})
        for sp, ui in zip(specs, row):
            lo, hi = sp.z_bounds()
            lo, hi = lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)
            s[sp.name] = sp.from_z(lo + ui * (hi - lo))
        out.append(s)
    return out


HESTON_HAND_STARTS = [  # deliberately different regimes
    dict(v0=0.004, kappa=0.5, theta=0.01, sigma=0.2, rho=0.3),
    dict(v0=0.02, kappa=5.0, theta=0.003, sigma=1.0, rho=-0.6),
    dict(v0=0.0064, kappa=1.5, theta=0.0064, sigma=0.4, rho=0.0),
]


@dataclass
class MultiStartResult:
    best: FitResult
    all: list[FitResult]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([f.summary() for f in self.all]).sort_values("cost").reset_index(drop=True)


def calibrate(spec: ModelSpec, data: CalibrationData, obj: Objective, starts: Sequence[Mapping[str, float]],
              fixed: Mapping[str, float] | None = None) -> MultiStartResult:
    fits = [fit(spec, data, obj, s, fixed) for s in starts]
    ok = [f for f in fits if f.n_invalid == 0] or fits
    return MultiStartResult(min(ok, key=lambda f: f.cost), fits)


@dataclass
class NestedCalibration:
    heston: MultiStartResult
    bates: MultiStartResult
    hbmj_events_only: MultiStartResult
    hbmj: MultiStartResult
    event_types_free: tuple[str, ...]
    event_types_unidentified: tuple[str, ...]

    def best(self) -> dict[str, FitResult]:
        return {"Heston": self.heston.best, "Bates": self.bates.best, "HB-MJ": self.hbmj.best}


JUMP_STARTS = [dict(lam=0.5, mu_J=0.0, sigma_J=0.02), dict(lam=3.0, mu_J=-0.01, sigma_J=0.01),
               dict(lam=0.2, mu_J=-0.03, sigma_J=0.03), dict(lam=1e-6, mu_J=0.0, sigma_J=0.01)]


def calibrate_nested(data: CalibrationData, obj: Objective = Objective(), n_random: int = 2, seed: int = 0,
                     heston_starts: Sequence[Mapping[str, float]] | None = None) -> NestedCalibration:
    """Heston -> Bates -> HB-MJ, each stage started from the previous one."""
    hs = list(heston_starts) if heston_starts is not None else HESTON_HAND_STARTS + random_starts(HESTON, n_random, seed)
    h = calibrate(HESTON, data, obj, hs)
    b = calibrate(BATES, data, obj, [{**h.best.params, **j} for j in JUMP_STARTS])
    free_ev = data.identifiable_event_types()
    unident = tuple(k for k in EVENT_TYPES if k not in free_ev)
    fixed_zero = {f"sigma_{k}": 0.0 for k in unident}
    ev_starts = [{f"sigma_{k}": s for k in EVENT_TYPES} for s in (0.002, 0.006, 0.012)]
    # 3a: only the event sigmas move; Heston/Bates parameters fixed at the Bates fit.
    e_only = calibrate(HBMJ, data, obj, [{**b.best.params, **s} for s in ev_starts],
                       fixed={**b.best.params, **fixed_zero})
    # 3b: joint refinement from the event-only solution (and from the Bates fit + mid event start).
    joint = calibrate(HBMJ, data, obj, [e_only.best.params, {**b.best.params, **ev_starts[1]}], fixed=fixed_zero)
    return NestedCalibration(h, b, e_only, joint, free_ev, unident)


# =========================================================================== #
# Identifiability
# =========================================================================== #

def identifiability(result: FitResult, data: CalibrationData, obj: Objective, noise_vol: float = 0.001,
                    corr_flag: float = 0.95, cond_flag: float = 1e8) -> dict:
    """Local identifiability at a fit, from the Jacobian of the weighted residuals.

    J is taken w.r.t. *natural* parameters scaled by a typical magnitude (so columns are
    comparable), by central differences. With IV quote noise of `noise_vol` (default 0.1 vol pt)
    the approximate covariance is noise^2 (J^T J)^-1 in natural units. Reported:
      se           standard error per free parameter under that noise
      corr         correlation matrix -> pairs with |corr| > corr_flag are flagged
      singular values of the scaled J, condition number, and the flattest direction
      zero columns -> a parameter that moves no residual at all (not identified)
    """
    spec = {"Heston": HESTON, "Bates": BATES, "HB-MJ": HBMJ}[result.model]
    names = list(result.free)
    x = dict(result.params)
    w = weights(data, obj)
    typ = np.array([max(SPECS_BY_NAME[n].typical(x[n]), 1e-8) for n in names])
    J = np.empty((len(data.iv), len(names)))
    for j, n in enumerate(names):
        h = 1e-4 * typ[j]
        sp = SPECS_BY_NAME[n]
        up, dn = min(x[n] + h, sp.upper), max(x[n] - h, sp.lower)
        r_up = evaluate(spec.build({**x, n: up}, data.events), data, obj, w).residuals
        r_dn = evaluate(spec.build({**x, n: dn}, data.events), data, obj, w).residuals
        J[:, j] = (r_up - r_dn) / (up - dn)
    Js = J * typ  # sensitivity to a 100% relative change (or one typical unit)
    col_norm = np.linalg.norm(Js, axis=0)
    zero_cols = [n for n, c in zip(names, col_norm) if c < 1e-10]
    U, s, Vt = np.linalg.svd(Js, full_matrices=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    # Tiny ridge, NOT a pseudo-inverse: pinv would drop an exactly flat direction and report only the
    # variance of the identified combination (e.g. sigma_FOMC^2 + sigma_ECB^2 -> spurious corr +1).
    # With the ridge the flat direction dominates: huge SEs and corr -1, i.e. "only the sum is known".
    JtJ = J.T @ J
    ridge = 1e-14 * max(float(np.max(np.diag(JtJ))), 1e-300)
    cov = noise_vol**2 * np.linalg.inv(JtJ + ridge * np.eye(len(names)))
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    d = np.where(se > 0, se, np.nan)
    corr = cov / np.outer(d, d)
    pairs = [(names[i], names[j], float(corr[i, j])) for i in range(len(names)) for j in range(i + 1, len(names))
             if np.isfinite(corr[i, j]) and abs(corr[i, j]) > corr_flag]
    flat = {n: float(v) for n, v in zip(names, Vt[-1])}
    flags = []
    if zero_cols:
        flags.append(f"not identified (no effect on any contract): {zero_cols}")
    if cond > cond_flag:
        flags.append(f"ill-conditioned: condition number {cond:.2e}")
    for a, b, c in pairs:
        flags.append(f"{a} ~ {b} substitute (corr {c:+.3f})")
    return {"names": names, "se": dict(zip(names, se)), "corr": pd.DataFrame(corr, index=names, columns=names),
            "singular_values": s, "condition_number": cond, "flattest_direction": flat,
            "zero_columns": zero_cols, "high_corr_pairs": pairs, "flags": flags}


def equivalent_fits(ms: MultiStartResult, rmse_tol: float = 1e-4, rel_param_tol: float = 0.05) -> list[dict]:
    """Starts that fit (almost) as well as the best but land on materially different parameters.

    Such pairs are an identifiability problem: the data cannot tell the parameter sets apart.
    rmse_tol is in decimal vol (default 0.01 vol pt).
    """
    best = ms.best
    out = []
    for f in ms.all:
        if f is best or f.iv_rmse - best.iv_rmse > rmse_tol or f.n_invalid:
            continue
        diffs = {k: (f.params[k] - best.params[k]) / max(abs(best.params[k]), 1e-6) for k in best.free}
        big = {k: v for k, v in diffs.items() if abs(v) > rel_param_tol}
        if big:
            out.append({"rmse_best_bp": 1e4 * best.iv_rmse, "rmse_other_bp": 1e4 * f.iv_rmse, "param_rel_diff": big})
    return out


# =========================================================================== #
# Metrics, model comparison and event breakdowns
# =========================================================================== #

def fit_metrics(result: FitResult, data: CalibrationData, obj: Objective, noise_floor_vol: float = 1e-4) -> dict:
    """IV RMSE/MAE (vol pts), price RMSE (pips of forward), weighted objective, AIC/BIC.

    AIC = n ln(RSS/n) + 2k,  BIC = n ln(RSS/n) + k ln n, with RSS the weighted squared IV
    residuals (Gaussian-error likelihood up to a constant). RSS/n is floored at
    noise_floor_vol^2 so noise-free synthetic fits do not produce -inf; k = free parameters.
    """
    spec = {"Heston": HESTON, "Bates": BATES, "HB-MJ": HBMJ}[result.model]
    ev = evaluate(spec.build(result.params, data.events), data, Objective("iv", obj.weighting, obj.spread_floor))
    n, k = len(data.iv), len(result.free)
    err = ev.iv_model - data.iv
    rss = float(np.sum(ev.residuals**2))
    ll = n * np.log(max(rss / n, noise_floor_vol**2))
    c = data.contracts
    return {"model": result.model, "n": n, "k": k, "iv_rmse_volpts": 100 * np.sqrt(np.mean(err**2)),
            "iv_mae_volpts": 100 * np.mean(np.abs(err)),
            "price_rmse_pips": 1e4 * np.sqrt(np.mean((ev.price_model - data.price) ** 2)),
            "weighted_obj": 0.5 * rss, "aic": ll + 2 * k, "bic": ll + k * np.log(n)}


def compare_models(fits: Mapping[str, FitResult], data: CalibrationData, obj: Objective) -> pd.DataFrame:
    return pd.DataFrame([fit_metrics(f, data, obj) for f in fits.values()])


def subset_masks(data: CalibrationData, window: float = 7 / 365) -> dict[str, np.ndarray]:
    """Named contract subsets for event-specific evaluation.

    before_<event_id> / after_<event_id>: expiries within `window` years before / after that
    event's time (before: not spanning it; after: spanning it).
    """
    T = data.contracts.T
    m = {"full": np.ones(len(T), bool), "event_spanning": data.event_spanning, "non_event": ~data.event_spanning}
    for k in EVENT_TYPES:
        m[f"{k}_spanning"] = data.spans_type(k)
    for e in data.events:
        if e.tau <= 0:
            continue
        before = (T < e.tau) & (T >= e.tau - window)
        after = (T >= e.tau) & (T <= e.tau + window)
        if before.any() and after.any():
            m[f"before_{e.event_id or e.kind}"] = before
            m[f"after_{e.event_id or e.kind}"] = after
    return m


def breakdown(fits: Mapping[str, FitResult], data: CalibrationData, window: float = 7 / 365) -> pd.DataFrame:
    """IV RMSE (vol pts) per model and contract subset -- same contracts for every model."""
    rows = []
    for name, f in fits.items():
        spec = {"Heston": HESTON, "Bates": BATES, "HB-MJ": HBMJ}[f.model]
        ev = evaluate(spec.build(f.params, data.events), data, Objective("iv", "equal"))
        err = ev.iv_model - data.iv
        for sub, mask in subset_masks(data, window).items():
            if mask.any():
                rows.append({"model": name, "subset": sub, "n": int(mask.sum()),
                             "iv_rmse_volpts": 100 * float(np.sqrt(np.mean(err[mask] ** 2))),
                             "iv_mae_volpts": 100 * float(np.mean(np.abs(err[mask])))})
    return pd.DataFrame(rows)


def variance_step(data: CalibrationData, iv: np.ndarray, event: ScheduledEvent) -> float | None:
    """ATM total-implied-variance step across an event: w(first expiry >= tau) - w(last expiry < tau).

    Uses the contract nearest the forward in each of the two expiries.
    """
    c = data.contracts
    Ts = np.unique(c.T)
    before, after = Ts[Ts < event.tau], Ts[Ts >= event.tau]
    if not len(before) or not len(after):
        return None

    def atm_w(t):
        m = np.where(c.T == t)[0]
        i = m[np.argmin(np.abs(np.log(c.K[m] / c.F[m])))]
        return iv[i] ** 2 * t

    return float(atm_w(after.min()) - atm_w(before.max()))


def verify_fit(result: FitResult, data: CalibrationData, n: int = 12, seed: int = 0) -> float:
    """Re-price a random subset with the independent Carr-Madan FFT; return max |price diff| / F."""
    spec = {"Heston": HESTON, "Bates": BATES, "HB-MJ": HBMJ}[result.model]
    model = spec.build(result.params, data.events)
    c = data.contracts
    idx = np.random.default_rng(seed).choice(len(c), size=min(n, len(c)), replace=False)
    a = price_european(model, c.F[idx], c.K[idx], c.T[idx], c.D[idx], c.is_call[idx])
    b = price_european(model, c.F[idx], c.K[idx], c.T[idx], c.D[idx], c.is_call[idx], method="fft")
    return float(np.max(np.abs(a - b) / c.F[idx]))

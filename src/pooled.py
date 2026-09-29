"""Pooled multi-date calibration: date-specific Heston/Bates state, event variances shared by type.

Why pool
--------
On a single valuation date the standard tenors (ON, 1W, 2W, 3W, 1M) rarely bracket each event,
so a single-date fit either cannot separate event variance from diffusion variance or lands in
a false optimum (the D3/D4 finding of the synthetic single-date study). Across a panel of
dates the calendar rolls: the ON option priced the day before an FOMC spans it, the one priced
the day after does not. One shared FOMC parameter must explain all of those contracts.

Parameters
----------
* local (per valuation date): a configurable subset of Heston/Bates parameters, by default
  v0, theta, rho -- the "state" of the vol surface; kappa, sigma and the Poisson-jump
  parameters are shared (structural). ``local=ALL_LOCAL`` makes every Heston/Bates parameter
  date-specific.
* shared event parameters, one per event *key*:
    - an event that is separated from its neighbours in the panel -> key = its type
      (sigma_FOMC, sigma_ECB, sigma_CPI, sigma_NFP), shared by all events of that type;
    - a run of events that NO contract in the panel can tell apart -> one combined parameter
      keyed by the sorted types, e.g. combo_ECB+FOMC, shared by all such runs.
  Two consecutive events a < b are *separated* if some contract spans one but not the other:
  a valuation time v with t_a <= v < t_b (a already public, b still ahead), or a contract
  valued before a that expires in [t_a, t_b). Both are checked with the same 0 < tau <= T rule
  used everywhere else.
  Sharing by type is an ASSUMPTION (the event variance of, say, FOMC meetings is constant over
  the panel). It is what lets a well-separated meeting inform a poorly separated one.

Optional smoothness penalty
---------------------------
``smooth > 0`` adds residual rows sqrt(smooth) * (z_{d,k} - z_{d-1,k}) for each local parameter k
in transformed units (log / atanh), i.e. a random-walk prior on the daily state. Off by default;
the study reports whether it is needed.

Engine
------
Single joint ``least_squares`` over [shared | local_1 | ... | local_D] with a block-sparse
Jacobian pattern (each date's residuals depend only on the shared block and its own local
block), so finite differences perturb all dates at once: the Jacobian costs about
(#local-per-date + #shared) residual evaluations regardless of the number of dates.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from src.calibration import (
    BOUND_TOL, INVALID_RESIDUAL, SPECS_BY_NAME, CalibrationData, Objective, ParamSpec, evaluate, weights,
)
from src.events import EVENT_TYPES, ScheduledEvent, select_events_before_expiry
from src.models.bates import PoissonJumps
from src.models.heston import HestonParams
from src.models.nested import NestedModel
from src.timeutils import to_utc

HESTON_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
JUMP_NAMES = ("lam", "mu_J", "sigma_J")
DEFAULT_LOCAL = ("v0", "theta", "rho")
ALL_LOCAL = HESTON_NAMES + JUMP_NAMES
COMBO_SPEC = lambda key: ParamSpec(key, 0.0, 0.05, "linear", 0.002)


# =========================================================================== #
# Panel data
# =========================================================================== #

@dataclass(frozen=True)
class CatalogEvent:
    event_id: str
    kind: str
    ts: pd.Timestamp


@dataclass(frozen=True)
class PanelDate:
    valuation_ts: pd.Timestamp
    data: CalibrationData   # its events carry tau relative to this valuation time
    label: str = ""


@dataclass(frozen=True)
class Panel:
    dates: tuple[PanelDate, ...]
    catalog: tuple[CatalogEvent, ...]

    @property
    def n_quotes(self) -> int:
        return sum(len(d.data.iv) for d in self.dates)


# =========================================================================== #
# Event clustering (what the panel can and cannot separate)
# =========================================================================== #

def _expiry_ts(pd_: PanelDate) -> list[pd.Timestamp]:
    v = to_utc(pd_.valuation_ts)
    return [v + pd.Timedelta(seconds=float(t) * 365 * 86400) for t in np.unique(pd_.data.contracts.T)]


def separated(a: CatalogEvent, b: CatalogEvent, panel: Panel) -> bool:
    """True if some contract in the panel spans exactly one of the two events (a before b)."""
    for d in panel.dates:
        v = to_utc(d.valuation_ts)
        if a.ts <= v < b.ts and any(x >= b.ts for x in _expiry_ts(d)):
            return True  # a already public, b ahead: this date's contracts span b but not a
        if v < a.ts and any(a.ts <= x < b.ts for x in _expiry_ts(d)):
            return True  # a contract spans a but expires before b
    return False


def event_clusters(panel: Panel) -> list[tuple[CatalogEvent, ...]]:
    """Maximal runs of consecutive events that no contract in the panel separates.

    Only events spanned by at least one contract are considered."""
    spanned = set()
    for d in panel.dates:
        for evs in d.data.spanned_events:
            spanned.update(e.event_id for e in evs)
    evs = sorted((e for e in panel.catalog if e.event_id in spanned), key=lambda e: e.ts)
    clusters: list[list[CatalogEvent]] = []
    for e in evs:
        if clusters and not separated(clusters[-1][-1], e, panel):
            clusters[-1].append(e)
        else:
            clusters.append([e])
    return [tuple(c) for c in clusters]


def cluster_key(cluster: Sequence[CatalogEvent]) -> str:
    if len(cluster) == 1:
        return f"sigma_{cluster[0].kind}"
    return "combo_" + "+".join(sorted(e.kind for e in cluster))


def event_parameter_map(panel: Panel) -> tuple[dict[str, str], list[str], list[tuple[CatalogEvent, ...]]]:
    """event_id -> parameter key (combined runs map their FIRST event to the combo key and the
    others to '' = zero variance; equivalent for every panel contract since none separates them)."""
    clusters = event_clusters(panel)
    mapping: dict[str, str] = {}
    for c in clusters:
        key = cluster_key(c)
        mapping[c[0].event_id] = key
        for e in c[1:]:
            mapping[e.event_id] = ""
    keys = sorted({k for k in mapping.values() if k})
    return mapping, keys, clusters


@dataclass(frozen=True)
class PerEventJumps:
    """Scheduled-event factor with one variance per event (a CFFactor)."""

    events: tuple[ScheduledEvent, ...]
    variance: Mapping[str, float]  # event_id -> sigma^2 (missing -> 0)

    def events_before(self, T: float) -> tuple[ScheduledEvent, ...]:
        return select_events_before_expiry(self.events, T)

    def cf_factor(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        v = sum(self.variance.get(e.event_id, 0.0) for e in self.events_before(T))
        return np.exp(-0.5 * v * u * (u + 1j))


# =========================================================================== #
# Pooled model specification
# =========================================================================== #

@dataclass(frozen=True)
class PooledSpec:
    model: str                         # "Heston" | "Bates" | "HB-MJ"
    local: tuple[str, ...]             # per-date parameters
    shared: tuple[str, ...]            # structural shared parameters (Heston/Bates part)
    event_keys: tuple[str, ...] = ()   # shared event parameters (HB-MJ only)

    @classmethod
    def make(cls, model: str, event_keys: Sequence[str] = (), local: Sequence[str] = DEFAULT_LOCAL) -> "PooledSpec":
        names = HESTON_NAMES + (JUMP_NAMES if model in ("Bates", "HB-MJ") else ())
        loc = tuple(n for n in names if n in local)
        sh = tuple(n for n in names if n not in local)
        return cls(model, loc, sh, tuple(event_keys) if model == "HB-MJ" else ())

    def spec(self, name: str) -> ParamSpec:
        return SPECS_BY_NAME[name] if name in SPECS_BY_NAME else COMBO_SPEC(name)

    @property
    def shared_all(self) -> tuple[str, ...]:
        return self.shared + self.event_keys

    def n_params(self, n_dates: int) -> int:
        return len(self.shared_all) + n_dates * len(self.local)


def build_date_model(spec: PooledSpec, shared: Mapping[str, float], local: Mapping[str, float],
                     events: tuple[ScheduledEvent, ...], mapping: Mapping[str, str]) -> NestedModel:
    x = {**shared, **local}
    h = HestonParams(x["kappa"], x["theta"], x["sigma"], x["rho"], x["v0"])
    jumps = PoissonJumps(x["lam"], x["mu_J"], x["sigma_J"]) if spec.model in ("Bates", "HB-MJ") else None
    ev = None
    if spec.model == "HB-MJ":
        var = {e.event_id: shared[mapping[e.event_id]] ** 2 for e in events if mapping.get(e.event_id)}
        ev = PerEventJumps(events, var)
    return NestedModel(h, jumps, ev)


# =========================================================================== #
# Fitting
# =========================================================================== #

@dataclass
class PooledFit:
    spec: PooledSpec
    shared: dict[str, float]
    local: list[dict[str, float]]
    mapping: dict[str, str]
    cost: float
    iv_rmse: float
    n_quotes: int
    n_eval: int
    status: int
    message: str
    success: bool
    seconds: float
    bound_hits: tuple[str, ...]
    jac: object = field(repr=False, default=None)
    residuals: np.ndarray = field(repr=False, default=None)
    names: list[str] = field(repr=False, default_factory=list)


class _Problem:
    def __init__(self, panel: Panel, spec: PooledSpec, obj: Objective, mapping: dict[str, str],
                 smooth: float, fixed_shared: Mapping[str, float] | None, threads: int):
        self.panel, self.spec, self.obj, self.mapping, self.smooth = panel, spec, obj, mapping, smooth
        self.fixed = dict(fixed_shared or {})
        self.shared_free = [n for n in spec.shared_all if n not in self.fixed]
        self.D, self.L = len(panel.dates), len(spec.local)
        self.w = [weights(d.data, obj) for d in panel.dates]
        self.n_res = [len(d.data.iv) for d in panel.dates]
        self.threads = threads
        specs = [spec.spec(n) for n in self.shared_free] + [spec.spec(n) for _ in range(self.D) for n in spec.local]
        self.specs = specs
        self.lb, self.ub = np.array([s.z_bounds() for s in specs]).T

    def names(self) -> list[str]:
        return self.shared_free + [f"{n}[{d}]" for d in range(self.D) for n in self.spec.local]

    def unpack(self, z: np.ndarray) -> tuple[dict, list[dict]]:
        S = len(self.shared_free)
        shared = {**self.fixed, **{n: s.from_z(zi) for n, s, zi in zip(self.shared_free, self.specs[:S], z[:S])}}
        local = []
        for d in range(self.D):
            blk = z[S + d * self.L: S + (d + 1) * self.L]
            local.append({n: SPECS_BY_NAME[n].from_z(zi) for n, zi in zip(self.spec.local, blk)})
        return shared, local

    def pack(self, shared: Mapping[str, float], local: Sequence[Mapping[str, float]]) -> np.ndarray:
        z = [s.to_z(s.clip(shared[n])) for n, s in zip(self.shared_free, self.specs)]
        for d in range(self.D):
            z += [SPECS_BY_NAME[n].to_z(SPECS_BY_NAME[n].clip(local[d][n])) for n in self.spec.local]
        return np.array(z)

    def _date_residuals(self, d: int, shared: dict, local: dict) -> np.ndarray:
        pdt = self.panel.dates[d]
        model = build_date_model(self.spec, shared, local, pdt.data.events, self.mapping)
        return evaluate(model, pdt.data, self.obj, self.w[d]).residuals

    def residuals(self, z: np.ndarray) -> np.ndarray:
        shared, local = self.unpack(z)
        if self.threads > 1:
            with ThreadPoolExecutor(self.threads) as ex:
                parts = list(ex.map(lambda d: self._date_residuals(d, shared, local[d]), range(self.D)))
        else:
            parts = [self._date_residuals(d, shared, local[d]) for d in range(self.D)]
        r = np.concatenate(parts)
        if self.smooth > 0 and self.L and self.D > 1:
            S = len(self.shared_free)
            zl = z[S:].reshape(self.D, self.L)
            r = np.concatenate([r, np.sqrt(self.smooth) * np.diff(zl, axis=0).ravel()])
        return r

    def sparsity(self):
        S = len(self.shared_free)
        n_rows = sum(self.n_res) + (self.L * (self.D - 1) if self.smooth > 0 and self.L and self.D > 1 else 0)
        J = lil_matrix((n_rows, S + self.D * self.L), dtype=int)
        row = 0
        for d, n in enumerate(self.n_res):
            J[row:row + n, :S] = 1
            J[row:row + n, S + d * self.L: S + (d + 1) * self.L] = 1
            row += n
        if n_rows > row:
            for d in range(1, self.D):
                for k in range(self.L):
                    J[row, S + (d - 1) * self.L + k] = 1
                    J[row, S + d * self.L + k] = 1
                    row += 1
        return J.tocsr()


class _Timeout(Exception):
    pass


class _Monitor:
    """Wraps the residual function: counts evaluations, keeps the best point seen, logs progress
    and enforces a wall-clock limit (least_squares has none)."""

    def __init__(self, fun, deadline: float | None, progress, label: str, every: int = 25):
        self.fun, self.deadline, self.progress, self.label, self.every = fun, deadline, progress, label, every
        self.n, self.best_cost, self.best_z, self.t0 = 0, np.inf, None, time.perf_counter()

    def __call__(self, z):
        # Always evaluate at least the starting point, so a best point exists when time runs out.
        if self.n > 0 and self.deadline is not None and time.perf_counter() > self.deadline:
            raise _Timeout
        r = self.fun(z)
        self.n += 1
        c = 0.5 * float(np.sum(r * r))
        if c < self.best_cost:
            self.best_cost, self.best_z = c, np.array(z, copy=True)
        if self.progress is not None and (self.n == 1 or self.n % self.every == 0):
            self.progress(f"{self.label}: {self.n} residual evals, best cost {self.best_cost:.6e}, "
                          f"{time.perf_counter() - self.t0:.0f}s")
        return r


def fit_pooled(panel: Panel, spec: PooledSpec, obj: Objective, shared0: Mapping[str, float],
               local0: Sequence[Mapping[str, float]], mapping: dict[str, str] | None = None,
               smooth: float = 0.0, fixed_shared: Mapping[str, float] | None = None,
               max_nfev: int = 300, threads: int = 8, timeout_s: float | None = None,
               progress=None, label: str = "") -> PooledFit:
    """Joint pooled least squares. With ``timeout_s``, a fit that runs out of time returns the best
    point evaluated so far with status -99 / success=False ("timed out") -- never reported as converged.
    ``progress`` (callable taking a string) receives a line every 25 residual evaluations."""
    mapping = mapping if mapping is not None else event_parameter_map(panel)[0]
    prob = _Problem(panel, spec, obj, mapping, smooth, fixed_shared, threads)
    z0 = prob.pack({**shared0, **(fixed_shared or {})}, local0)
    t0 = time.perf_counter()
    mon = _Monitor(prob.residuals, t0 + timeout_s if timeout_s else None, progress, label or spec.model)
    try:
        res = least_squares(mon, z0, jac_sparsity=prob.sparsity(), bounds=(prob.lb, prob.ub), method="trf",
                            tr_solver="lsmr", x_scale="jac", max_nfev=max_nfev, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    except _Timeout:
        res = type("Res", (), dict(x=mon.best_z, nfev=mon.n, status=-99, success=False, jac=None,
                                   message=f"timed out after {timeout_s:.0f}s (best point so far)"))()
        if progress is not None:
            progress(f"{label or spec.model}: TIMEOUT after {mon.n} evals, best cost {mon.best_cost:.6e}")
    shared, local = prob.unpack(res.x)
    r_fit = np.concatenate([prob._date_residuals(d, shared, local[d]) for d in range(prob.D)])
    names = prob.names()
    hits = tuple(n for n, zi, lo, hi in zip(names, res.x, prob.lb, prob.ub)
                 if min(zi - lo, hi - zi) < BOUND_TOL * (hi - lo) and not n.endswith("]"))
    # unweighted IV RMSE over all quotes
    errs = []
    for d, pdt in enumerate(panel.dates):
        m = build_date_model(spec, shared, local[d], pdt.data.events, mapping)
        errs.append(evaluate(m, pdt.data, Objective("iv", "equal")).residuals)
    e = np.concatenate(errs)
    return PooledFit(spec, shared, local, dict(mapping), float(0.5 * np.sum(r_fit**2)),
                     float(np.sqrt(np.mean(e**2))), panel.n_quotes, int(res.nfev), int(res.status), str(res.message),
                     bool(res.success), time.perf_counter() - t0, hits, res.jac, r_fit, names)


def heuristic_local_start(panel: Panel, spec: PooledSpec) -> list[dict[str, float]]:
    """Per-date start from the surface itself: v0 ~ shortest ATM IV^2, theta ~ longest ATM IV^2."""
    out = []
    for pdt in panel.dates:
        c, iv = pdt.data.contracts, pdt.data.iv
        atm = lambda t: iv[np.where(c.T == t)[0][np.argmin(np.abs(np.log(c.K[c.T == t] / c.F[c.T == t])))]]
        Ts = np.unique(c.T)
        base = {"v0": atm(Ts[0]) ** 2, "theta": atm(Ts[-1]) ** 2, "rho": 0.0, "kappa": 2.0, "sigma": 0.4,
                "lam": 0.5, "mu_J": 0.0, "sigma_J": 0.01}
        out.append({n: base[n] for n in spec.local})
    return out


@dataclass
class PooledNested:
    heston: PooledFit
    bates: PooledFit
    hbmj_events_only: PooledFit
    hbmj: PooledFit
    event_keys: tuple[str, ...]
    clusters: list

    def best(self) -> dict[str, PooledFit]:
        return {"Heston": self.heston, "Bates": self.bates, "HB-MJ": self.hbmj}


def restore_fit(panel: Panel, spec: PooledSpec, shared: Mapping[str, float], local: Sequence[Mapping[str, float]],
                mapping: dict[str, str], obj: Objective, status: int, success: bool, message: str) -> PooledFit:
    """Rebuild a PooledFit from checkpointed parameters (one residual evaluation, no optimisation).

    Used when resuming: a completed stage is not re-optimised; its saved parameters seed the next stage.
    The Jacobian is not stored, so identifiability diagnostics need the stage refit (they are optional)."""
    prob = _Problem(panel, spec, obj, mapping, 0.0, None, 1)
    shared, local = dict(shared), [dict(l) for l in local]
    r = np.concatenate([prob._date_residuals(d, shared, local[d]) for d in range(prob.D)])
    e = np.concatenate([evaluate(build_date_model(spec, shared, local[d], pdt.data.events, mapping), pdt.data,
                                 Objective("iv", "equal")).residuals for d, pdt in enumerate(panel.dates)])
    return PooledFit(spec, shared, local, dict(mapping), float(0.5 * np.sum(r**2)), float(np.sqrt(np.mean(e**2))),
                     panel.n_quotes, 0, status, f"restored from checkpoint: {message}", success, 0.0, (), None, r,
                     prob.names())


def calibrate_pooled_nested(panel: Panel, obj: Objective = Objective(), local: Sequence[str] = DEFAULT_LOCAL,
                            smooth: float = 0.0, threads: int = 8, shared_starts: Sequence[Mapping] | None = None,
                            timeout_s: float | None = None, progress=None, on_stage=None,
                            done: Mapping[str, Mapping] | None = None) -> PooledNested:
    """Pooled Heston -> Bates -> HB-MJ, each stage started from the previous one.

    timeout_s / progress are passed to every fit (see fit_pooled).
    on_stage(model_name, fit): called as soon as each model stage finishes (use it to checkpoint).
    done: {model_name: {"shared", "local", "status", "success", "message"}} of stages already completed;
          they are restored from their saved parameters instead of being re-optimised.
    """
    kw = dict(threads=threads, timeout_s=timeout_s, progress=progress)
    done = dict(done or {})
    mapping, keys, clusters = event_parameter_map(panel)

    def stage(name, spec, run):
        if name in done:
            d = done[name]
            f = restore_fit(panel, spec, d["shared"], d["local"], mapping, obj, d["status"], d["success"], d["message"])
            if progress is not None:
                progress(f"{name}: restored from checkpoint (not re-optimised)")
            return f
        f = run()
        if on_stage is not None:
            on_stage(name, f)
        return f

    hs = PooledSpec.make("Heston", local=local)
    loc0 = heuristic_local_start(panel, PooledSpec.make("HB-MJ", keys, local=ALL_LOCAL))
    starts = shared_starts or [dict(kappa=2.0, sigma=0.4, rho=0.0, v0=0.005, theta=0.007),
                               dict(kappa=0.8, sigma=0.8, rho=-0.3, v0=0.005, theta=0.007)]
    h = stage("Heston", hs, lambda: min(
        (fit_pooled(panel, hs, obj, s, [{n: l[n] for n in hs.local} for l in loc0], mapping, smooth,
                    label=f"Heston start {i}", **kw) for i, s in enumerate(starts)), key=lambda f: f.cost))
    bs = PooledSpec.make("Bates", local=local)

    def run_bates():
        fits = []
        for i, j in enumerate((dict(lam=0.5, mu_J=0.0, sigma_J=0.02), dict(lam=3.0, mu_J=-0.01, sigma_J=0.01))):
            sh0 = {**h.shared, **{k: v for k, v in j.items() if k in bs.shared}}
            l0 = [{**l, **{k: v for k, v in j.items() if k in bs.local}} for l in h.local]
            fits.append(fit_pooled(panel, bs, obj, sh0, l0, mapping, smooth, label=f"Bates start {i}", **kw))
        return min(fits, key=lambda f: f.cost)

    b = stage("Bates", bs, run_bates)
    es = PooledSpec.make("HB-MJ", keys, local=local)
    ev0 = {k: 0.004 for k in keys}
    holder = {}

    def run_hbmj():
        # 3a: event parameters only (Heston/Bates part fixed at the pooled Bates fit); 3b: joint
        e_only = _events_only(panel, es, obj, b, ev0, mapping, threads)
        holder["e_only"] = e_only
        if progress is not None:
            progress(f"HB-MJ events-only stage done: cost {e_only.cost:.6e}")
        return fit_pooled(panel, es, obj, e_only.shared, b.local, mapping, smooth, label="HB-MJ joint", **kw)

    joint = stage("HB-MJ", es, run_hbmj)
    return PooledNested(h, b, holder.get("e_only"), joint, tuple(keys), clusters)


def _events_only(panel, spec, obj, bates_fit, ev0, mapping, threads) -> PooledFit:
    """Fit only the shared event variances, all Heston/Bates parameters (shared and local) fixed."""
    # Each date's Bates parameters are held at the pooled Bates fit inside the model build.
    class _Frozen(_Problem):
        def _date_residuals(self_, d, shared, local):
            pdt = self_.panel.dates[d]
            model = build_date_model(spec, {**bates_fit.shared, **{k: shared[k] for k in spec.event_keys}},
                                     bates_fit.local[d], pdt.data.events, mapping)
            return evaluate(model, pdt.data, self_.obj, self_.w[d]).residuals

    ev_spec = PooledSpec(spec.model, (), (), spec.event_keys)
    prob = _Frozen(panel, ev_spec, obj, mapping, 0.0, None, threads)
    z0 = prob.pack(ev0, [{} for _ in panel.dates])
    t0 = time.perf_counter()
    res = least_squares(prob.residuals, z0, bounds=(prob.lb, prob.ub), method="trf", x_scale="jac", max_nfev=200,
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    shared_ev, _ = prob.unpack(res.x)
    r = prob.residuals(res.x)
    return PooledFit(spec, {**bates_fit.shared, **shared_ev}, bates_fit.local, dict(mapping), float(0.5 * np.sum(r**2)),
                     np.nan, panel.n_quotes, int(res.nfev), int(res.status), str(res.message), bool(res.success),
                     time.perf_counter() - t0, (), None, r, prob.names())


# =========================================================================== #
# Diagnostics, comparison and out-of-sample
# =========================================================================== #

def pooled_identifiability(fit: PooledFit, noise_vol: float = 0.001) -> dict:
    """SEs and correlations of the SHARED parameters, with all local parameters profiled out
    (full-Jacobian covariance, tiny ridge -- see calibration.identifiability)."""
    if fit.jac is None:
        raise ValueError("fit has no Jacobian")
    J = fit.jac.toarray() if hasattr(fit.jac, "toarray") else np.asarray(fit.jac)
    J = J[: fit.n_quotes]  # drop smoothness rows
    JtJ = J.T @ J
    ridge = 1e-14 * max(float(np.max(np.diag(JtJ))), 1e-300)
    cov_z = noise_vol**2 * np.linalg.inv(JtJ + ridge * np.eye(JtJ.shape[0]))
    shared_names = [n for n in fit.names if not n.endswith("]")]
    idx = [fit.names.index(n) for n in shared_names]
    # delta method: z -> x for shared parameters
    dxdz = np.array([_dxdz(fit.spec.spec(n), fit.shared[n]) for n in shared_names])
    cov = cov_z[np.ix_(idx, idx)] * np.outer(dxdz, dxdz)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    d = np.where(se > 0, se, np.nan)
    corr = pd.DataFrame(cov / np.outer(d, d), index=shared_names, columns=shared_names)
    ev = np.linalg.eigvalsh(JtJ)
    return {"se": dict(zip(shared_names, se)), "corr": corr,
            "condition_number": float(ev.max() / max(ev.min(), 1e-300))}


def _dxdz(spec: ParamSpec, x: float) -> float:
    if spec.transform == "log":
        return x
    if spec.transform == "atanh":
        return 1 - x * x
    return spec.scale


def pooled_metrics(fit: PooledFit, panel: Panel, noise_floor_vol: float = 1e-4) -> dict:
    n, k = panel.n_quotes, fit.spec.n_params(len(panel.dates))
    rss = float(np.sum(fit.residuals**2))
    ll = n * np.log(max(rss / n, noise_floor_vol**2))
    return {"model": fit.spec.model, "n": n, "k": k, "iv_rmse_volpts": 100 * fit.iv_rmse, "weighted_obj": 0.5 * rss,
            "aic": ll + 2 * k, "bic": ll + k * np.log(n), "success": fit.success, "status": fit.status}


def holdout_error(fit: PooledFit, panel: Panel, heldout: Sequence[CalibrationData]) -> dict:
    """IV RMSE on held-out contracts (one CalibrationData per panel date, same valuation) using that
    date's fitted local parameters and the shared parameters. Split into event-spanning vs not."""
    all_e, spans = [], []
    for d, data in enumerate(heldout):
        if len(data.iv) == 0:
            continue
        m = build_date_model(fit.spec, fit.shared, fit.local[d], data.events, fit.mapping)
        all_e.append(evaluate(m, data, Objective("iv", "equal")).residuals)
        spans.append(data.event_spanning)
    e, s = np.concatenate(all_e), np.concatenate(spans)
    f = lambda m: 100 * float(np.sqrt(np.mean(e[m] ** 2))) if m.any() else np.nan
    return {"model": fit.spec.model, "oos_iv_rmse_volpts": f(np.ones(len(e), bool)),
            "oos_event_spanning": f(s), "oos_non_event": f(~s), "n_oos": len(e)}

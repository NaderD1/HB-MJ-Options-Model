"""Synthetic daily panels for testing the pooled calibration.

Valuation at 16:00 New York on each EUR/USD joint business day; standard tenors (default ON,
1W, 2W, 3W, 1M) with expiry dates from the FX date roller and a 10:00 New York cut. Tenors
whose expiry the roller reports as ambiguous (e.g. 1M across Thanksgiving) are DROPPED for
that date -- no expiry is guessed. Events come from the as-known calendar (or a supplied
catalog); the true event variance is shared by event type.

Daily state (truth): log v0 follows an AR(1) around log(v_bar); log theta and rho follow small
random walks; kappa, sigma and the Poisson-jump parameters are constant.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.calibration import CalibrationData, synthetic_surface
from src.events import EventCalendar, ScheduledEvent
from src.fx_calendar import is_business_day, option_dates
from src.metrics import ContractSet
from src.models.bates import PoissonJumps
from src.models.black_scholes import black76_price, implied_vol_fast
from src.models.heston import HestonParams
from src.models.nested import NestedModel
from src.pooled import CatalogEvent, Panel, PanelDate, PerEventJumps
from src.pricing_engine import price_european
from src.timeutils import localize, year_fraction

TENORS = ("ON", "1W", "2W", "3W", "1M")
Z_GRID = (-1.8, -0.95, 0.0, 0.95, 1.8)


@dataclass(frozen=True)
class PanelTruth:
    shared: dict                   # kappa, sigma, lam, mu_J, sigma_J
    event_sigma: dict              # type -> sigma (e.g. {"FOMC": 0.006, ...})
    daily: list[dict]              # per date: v0, theta, rho


def business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    out, d = [], start
    while d <= end:
        if is_business_day(d, "EURUSD"):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


def catalog_from_calendar(start: pd.Timestamp, end: pd.Timestamp, calendar: EventCalendar | None = None) -> tuple[CatalogEvent, ...]:
    cal = calendar or EventCalendar.from_csv()
    a = cal.active
    a = a[(a.timestamp_utc >= start) & (a.timestamp_utc <= end)]
    return tuple(CatalogEvent(i, k, t) for i, k, t in zip(a.event_id, a.event_type, a.timestamp_utc))


def simulate_state(n: int, seed: int, v_bar: float = 0.0049, theta0: float = 0.0081, rho0: float = -0.25,
                   phi: float = 0.9, sd_v: float = 0.08, sd_theta: float = 0.02, sd_rho: float = 0.01) -> list[dict]:
    rng = np.random.default_rng(seed)
    lv, lt, r = np.log(v_bar), np.log(theta0), rho0
    out = []
    for _ in range(n):
        lv = np.log(v_bar) + phi * (lv - np.log(v_bar)) + sd_v * rng.standard_normal()
        lt += sd_theta * rng.standard_normal()
        r = float(np.clip(r + sd_rho * rng.standard_normal(), -0.9, 0.9))
        out.append({"v0": float(np.exp(lv)), "theta": float(np.exp(lt)), "rho": r})
    return out


def make_panel(dates: Sequence[dt.date], catalog: tuple[CatalogEvent, ...], truth: PanelTruth,
               tenors: Sequence[str] = TENORS, z_grid: Sequence[float] = Z_GRID, noise_vol: float = 0.0,
               seed: int = 0, spot: float = 1.16, r_usd: float = 0.038, r_eur: float = 0.021,
               ) -> tuple[Panel, list[dict]]:
    """Panel priced by the true model. Returns (panel, per-date info incl. dropped tenors)."""
    rng = np.random.default_rng(seed)
    pdates, info = [], []
    for i, d in enumerate(dates):
        v = localize(d.isoformat(), "16:00", "America/New_York")
        exps, kept, dropped = [], [], []
        for tn in tenors:
            od = option_dates(d, tn)
            if od["expiry"] is None:
                dropped.append(tn)
                continue
            exps.append(year_fraction(v, localize(od["expiry"].isoformat(), "10:00", "America/New_York")))
            kept.append(tn)
        events = tuple(ScheduledEvent(year_fraction(v, e.ts), e.kind, e.event_id) for e in catalog)
        x = {**truth.shared, **truth.daily[i]}
        model = NestedModel(HestonParams(x["kappa"], x["theta"], x["sigma"], x["rho"], x["v0"]),
                            PoissonJumps(x["lam"], x["mu_J"], x["sigma_J"]),
                            PerEventJumps(events, {e.event_id: truth.event_sigma.get(e.kind, 0.0) ** 2 for e in events}))
        T = np.repeat(np.asarray(exps), len(z_grid))
        z = np.tile(np.asarray(z_grid), len(exps))
        F, D = spot * np.exp((r_usd - r_eur) * T), np.exp(-r_usd * T)
        K = F * np.exp(z * 0.08 * np.sqrt(T))
        C = K >= F
        price = price_european(model, F, K, T, D, C)
        iv, ok = implied_vol_fast(price, F, K, T, D, C, guess=np.full(T.shape, 0.08))
        assert ok.all(), f"IV inversion failed on {d}"
        if noise_vol > 0:
            iv = iv + rng.normal(0.0, noise_vol, iv.shape)
            price = black76_price(F, K, T, iv, D, C)
        labels = tuple(np.repeat(np.asarray(kept), len(z_grid)))
        data = CalibrationData(ContractSet.from_arrays(F, K, T, D, C), iv, price, events,
                               iv - 0.002, iv + 0.002, labels)
        pdates.append(PanelDate(v, data, d.isoformat()))
        info.append({"date": d, "kept": kept, "dropped": dropped})
    return Panel(tuple(pdates), catalog), info


def split_heldout(panel: Panel, tenor: str) -> tuple[Panel, list[CalibrationData]]:
    """Leave one tenor out on every date: (panel without it, held-out contracts per date)."""
    kept, held = [], []
    for pdt in panel.dates:
        lab = np.asarray(pdt.data.labels)
        m = lab != tenor
        kept.append(PanelDate(pdt.valuation_ts, pdt.data.subset(m), pdt.label))
        held.append(pdt.data.subset(~m))
    return Panel(tuple(kept), panel.catalog), held

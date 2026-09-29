"""HB-MJ: Heston–Bates + scheduled-time, stochastic-size macro-event jumps.

Nesting (each model reuses the previous one and switches on one more factor):

    Heston = diffusion only
    Bates  = Heston x Poisson jumps          (src/models/bates.py)
    HB-MJ  = Bates  x scheduled-event jumps  (this module)

Scheduled-event component. The *timing* of each event is known in advance (a calendar
date for an FOMC, ECB, US CPI or NFP announcement), but the *size* of the move it causes
is random. At event time tau_i the log-spot jumps by

    Y_i ~ N(-sigma_E^2 / 2, sigma_E^2),     sigma_E = sigma_E[type of event i]

The mean -sigma_E^2/2 is the compensator: E[exp(Y_i)] = exp(-sigma_E^2/2 + sigma_E^2/2) = 1,
so each event leaves the forward unchanged on its own. Its CF is that of a normal variable:

    E[exp(i u Y_i)] = exp(i u (-sigma_E^2/2) - sigma_E^2 u^2 / 2) = exp(-sigma_E^2 u (u + i) / 2)

Only events with 0 < tau_i <= T affect an option expiring at T. Events are independent of
each other and of the Heston/Bates dynamics, so their factors multiply:

    phi_E(u; T) = prod_{i: 0 < tau_i <= T} exp(-sigma_{E,i}^2 u (u + i) / 2)
    phi_HBMJ    = phi_Heston * phi_PoissonJumps * phi_E

Checks: phi_E(0) = 1 and phi_E(-i) = exp(-sigma^2 (-i)(0)/2) = 1.

Important property (see tests): because each Y_i is *normal* and independent, for a single
European expiry the event component adds exactly sigma_E^2 of variance and nothing else --
no extra skew and no extra kurtosis of its own. For European options it is therefore
equivalent to a deterministic burst of extra Gaussian variance on the event date. Its
empirical fingerprint is a *step in the term structure of total implied variance* at
event dates, not a change in smile shape. The jump nature matters for path-dependent
quantities -- above all delta hedging across the event (a gap move cannot be hedged
continuously), which the hedging experiment is designed to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.events import EVENT_TYPES, ScheduledEvent, select_events_before_expiry
from src.models.bates import PoissonJumps
from src.models.heston import HestonParams
from src.models.nested import NestedModel

__all__ = ["EVENT_TYPES", "ScheduledEvent", "ScheduledEventJumps", "event_cf", "hbmj"]


def event_cf(u: np.ndarray, sigma_E: float) -> np.ndarray:
    """CF of one compensated normal log-jump: exp(-sigma_E^2 u (u + i) / 2)."""
    u = np.asarray(u, dtype=complex)
    return np.exp(-0.5 * sigma_E**2 * u * (u + 1j))


@dataclass(frozen=True)
class ScheduledEventJumps:
    """Scheduled-time, stochastic-size jump factor (a :class:`~src.models.nested.CFFactor`)."""

    events: tuple[ScheduledEvent, ...]
    sigma_E: Mapping[str, float] = field(default_factory=dict)  # event type -> jump std dev

    def __post_init__(self) -> None:
        for e in self.events:
            if e.kind not in self.sigma_E:
                raise ValueError(f"No sigma_E given for event type {e.kind!r}")
        if any(s < 0 for s in self.sigma_E.values()):
            raise ValueError("sigma_E must be non-negative")

    def events_before(self, T: float) -> tuple[ScheduledEvent, ...]:
        """Events that an option expiring at T is exposed to (0 < tau <= T).

        Delegates to src.events.select_events_before_expiry -- the single rule shared with
        the data layer's event-spanning tags.
        """
        return select_events_before_expiry(self.events, T)

    def event_variance(self, T: float) -> float:
        """Total log-variance added by scheduled events before T: sum of sigma_E^2."""
        return float(sum(self.sigma_E[e.kind] ** 2 for e in self.events_before(T)))

    def cf_factor(self, u: np.ndarray, T: float) -> np.ndarray:
        out = np.ones_like(np.asarray(u, dtype=complex))
        for e in self.events_before(T):
            out = out * event_cf(u, self.sigma_E[e.kind])
        return out


def hbmj(heston: HestonParams, jumps: PoissonJumps, events: ScheduledEventJumps) -> NestedModel:
    """HB-MJ = the given Bates components (Heston + Poisson jumps) with the event factor switched on."""
    return NestedModel(diffusion=heston, jumps=jumps, events=events)

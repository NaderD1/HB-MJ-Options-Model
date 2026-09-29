"""Nested model: GK ⊂ Heston ⊂ Bates ⊂ HB-MJ as one object with switchable components.

The log-return to expiry is a sum of *independent* pieces,

    X_T = X^diff_T  +  (compound-Poisson jumps)  +  (scheduled-event jumps before T),

and the CF of a sum of independent variables is the product of their CFs:

    phi_model(u) = phi_diffusion(u) * phi_jumps(u) * phi_events(u).

So each richer model *reuses* the previous one and multiplies by one more factor:

    GK      = GKParams diffusion
    Heston  = HestonParams diffusion
    Bates   = Heston * Poisson-jump factor       (Step 4)
    HB-MJ   = Bates  * scheduled-event factor    (Step 5)

Design invariant: every factor must satisfy factor(-i; T) = 1 on its own (its jumps are
"compensated" to have zero mean effect on S_T/F). Then any combination of switched-on
factors is automatically risk-neutral, and switching a factor off (None) recovers the
smaller model *exactly* -- which is what makes the nested comparison fair.

Independence is an assumption, not a free lunch: it rules out, e.g., event jumps that are
bigger when variance is high. That is a testable limitation for the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from src.models.black_scholes import GKParams
from src.models.heston import HestonParams


class CFFactor(Protocol):
    """A multiplicative CF component (jumps, events). Must equal 1 at u = -i."""

    def cf_factor(self, u: np.ndarray, T: float) -> np.ndarray: ...


@dataclass(frozen=True)
class NestedModel:
    diffusion: GKParams | HestonParams
    jumps: CFFactor | None = None   # Bates Poisson jumps; None = switched off
    events: CFFactor | None = None  # HB-MJ scheduled-time, stochastic-size jumps; None = off

    def cf(self, u: np.ndarray, T: float) -> np.ndarray:
        out = self.diffusion.cf(u, T)
        for factor in (self.jumps, self.events):
            if factor is not None:
                out = out * factor.cf_factor(u, T)
        return out

    def without(self, *components: str) -> "NestedModel":
        """Switch components off, e.g. ``hbmj.without("events")`` -> the nested Bates model."""
        return replace(self, **{c: None for c in components})

    @property
    def name(self) -> str:
        if isinstance(self.diffusion, GKParams):
            base = "GK"
        else:
            base = "Heston"
        if self.jumps is not None and self.events is not None:
            return "HB-MJ" if base == "Heston" else f"{base}+jumps+events"
        if self.jumps is not None:
            return "Bates" if base == "Heston" else f"{base}+jumps"
        if self.events is not None:
            return f"{base}+events"
        return base


def martingale_error(model: NestedModel | CFFactor, T: float) -> float:
    """|phi(-i) - 1| (or |factor(-i) - 1|): should be ~1e-15 for a correctly compensated component."""
    minus_i = np.array([-1j])
    val = model.cf(minus_i, T) if hasattr(model, "cf") else model.cf_factor(minus_i, T)
    return float(np.abs(val[0] - 1.0))

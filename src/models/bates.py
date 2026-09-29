"""Bates (1996): Heston stochastic volatility + compensated compound-Poisson jumps.

Bates is *not* a separate model here. It is the existing Heston diffusion multiplied by
one jump factor inside :class:`~src.models.nested.NestedModel`:

    phi_Bates(u; T) = phi_Heston(u; T) * phi_J(u; T)

Jump component. Jumps arrive as a Poisson process with intensity ``lam`` (expected jumps
per year). Each jump multiplies spot by e^J with J ~ N(mu_J, sigma_J^2), so the average
proportional jump is

    kbar = E[e^J] - 1 = exp(mu_J + sigma_J^2 / 2) - 1.

Left alone, jumps would change E[S_T]; to keep S/F a martingale the drift is lowered by
lam * kbar (the "compensator"). The log-return contribution over [0, T] is then

    X^J_T = sum_{n=1}^{N_T} J_n  -  lam * kbar * T,        N_T ~ Poisson(lam T)

and, because the sum of a Poisson number of iid terms has CF exp(lam T (E[e^{iuJ}] - 1)),

    phi_J(u; T) = exp( lam T [ exp(i u mu_J - sigma_J^2 u^2 / 2) - 1 - i u kbar ] ).

Checks built into the formula: phi_J(0) = 1 (probabilities sum to one) and
phi_J(-i) = exp(lam T [ (1 + kbar) - 1 - kbar ]) = 1 (forward martingale).

The jump factor adds a fixed amount of variance per unit time, lam (mu_J^2 + sigma_J^2),
but its contribution to excess kurtosis scales like 1/T. Heston generates tail thickness
through *changes* in variance, which take time to accumulate; for moderate vol-of-vol its
short-dated smile is therefore mild (in our tested parameterization the 1W butterfly was
~0.3 vol pts vs ~2.3 at 1Y). Heston can produce steep short-dated smiles, but typically
only with very large vol-of-vol -- the tension Bates (1996) documents between option-implied
and time-series behaviour of volatility. Jumps supply short-maturity kurtosis directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.heston import HestonParams
from src.models.nested import NestedModel


@dataclass(frozen=True)
class PoissonJumps:
    """Compensated compound-Poisson factor with lognormal jump sizes (a :class:`CFFactor`)."""

    lam: float      # jump intensity: expected number of jumps per year
    mu_J: float     # mean log jump size (negative -> EUR crash risk, positive -> EUR spike risk)
    sigma_J: float  # std dev of log jump size (0 -> every jump is exactly mu_J)

    NAMES = ("lam", "mu_J", "sigma_J")

    @property
    def kbar(self) -> float:
        """Mean proportional jump E[e^J] - 1."""
        return float(np.expm1(self.mu_J + 0.5 * self.sigma_J**2))

    def cf_factor(self, u: np.ndarray, T: float) -> np.ndarray:
        u = np.asarray(u, dtype=complex)
        jump_cf = np.exp(1j * u * self.mu_J - 0.5 * self.sigma_J**2 * u * u)
        return np.exp(self.lam * T * (jump_cf - 1.0 - 1j * u * self.kbar))

    def variance_rate(self) -> float:
        """Annual variance added by jumps: lam * E[J^2] = lam (mu_J^2 + sigma_J^2)."""
        return self.lam * (self.mu_J**2 + self.sigma_J**2)


def bates(heston: HestonParams, lam: float, mu_J: float, sigma_J: float) -> NestedModel:
    """Bates model = the given Heston component with a Poisson jump factor switched on."""
    return NestedModel(diffusion=heston, jumps=PoissonJumps(lam, mu_J, sigma_J))

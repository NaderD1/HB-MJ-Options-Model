"""Model comparison on a fixed set of contracts.

Rule for every fit comparison in this project: all models are priced on the *same*
contract coordinates (F, K, T, D, call/put) -- the contracts that actually trade. Strikes
are never re-scaled per model here. Model-specific strike grids (e.g. "+-1 sd of each
model's own ATM vol") are allowed only in diagnostic plots, never in error statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.models.black_scholes import implied_vol_array
from src.pricing_engine import Model, price_european


@dataclass(frozen=True)
class ContractSet:
    """Immutable contract coordinates shared by all models in a comparison."""

    F: np.ndarray
    K: np.ndarray
    T: np.ndarray
    D: np.ndarray
    is_call: np.ndarray

    @classmethod
    def from_arrays(cls, F: ArrayLike, K: ArrayLike, T: ArrayLike, D: ArrayLike, is_call: ArrayLike) -> "ContractSet":
        arrs = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (F, K, T, D)), np.asarray(is_call, dtype=bool))
        frozen = []
        for a in arrs:
            a = np.array(a.ravel())  # own, contiguous copy
            a.setflags(write=False)
            frozen.append(a)
        return cls(*frozen)

    def __len__(self) -> int:
        return len(self.K)


def evaluate_models(
    models: Mapping[str, Model], contracts: ContractSet, method: str = "integration"
) -> pd.DataFrame:
    """Price and invert IV for every model on the identical contract set (long format)."""
    rows = []
    c = contracts
    for name, model in models.items():
        price = price_european(model, c.F, c.K, c.T, c.D, c.is_call, method=method)
        iv, status = implied_vol_array(price, c.F, c.K, c.T, c.D, c.is_call)
        rows.append(
            pd.DataFrame(
                {"contract_id": np.arange(len(c)), "model": name, "F": c.F, "K": c.K, "T": c.T,
                 "D": c.D, "is_call": c.is_call, "price": price, "iv": iv, "iv_status": status}
            )
        )
    return pd.concat(rows, ignore_index=True)

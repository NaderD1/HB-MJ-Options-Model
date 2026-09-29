"""Helpers shared by the pooled synthetic study and its tests."""

from __future__ import annotations

import numpy as np

from src.panel_synthetic import PanelTruth


def true_shared(truth: PanelTruth, keys) -> dict:
    """True values of the pooled shared parameters: structural + event keys (combos = sqrt of summed variance)."""
    out = dict(truth.shared)
    for k in keys:
        if k.startswith("sigma_"):
            out[k] = truth.event_sigma.get(k[6:], 0.0)
        elif k.startswith("combo_"):
            out[k] = float(np.sqrt(sum(truth.event_sigma.get(t, 0.0) ** 2 for t in k[6:].split("+"))))
    return out

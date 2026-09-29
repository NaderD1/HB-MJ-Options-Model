"""The external-validation comparison must pass when our numbers equal Bloomberg's and flag gaps."""

from pathlib import Path

import pandas as pd

from src.bloomberg_validation import BUCKETS, compare_with_bloomberg
from src.data_loaders import load_bloomberg_csv

EXAMPLE = Path(__file__).resolve().parents[1] / "data" / "examples" / "bloomberg_format_SYNTHETIC_eurusd.csv"


def _filled_table(tmp_path, perturb=None):
    raw = pd.read_csv(EXAMPLE)
    norm = load_bloomberg_csv(EXAMPLE)
    for b in BUCKETS:
        sub = norm[norm.bucket == b].set_index("tenor")
        raw[f"bbg_K_{b}"] = raw.tenor.map(sub.strike)
        raw[f"bbg_vol_{b}"] = raw.tenor.map(100 * sub.iv_mid)
    raw["bbg_premium_ATM_call_usd_pips"] = raw.tenor.map(1e4 * norm[norm.bucket == "ATM"].set_index("tenor").price_mid)
    if perturb:
        perturb(raw)
    p = tmp_path / "filled.csv"
    raw.to_csv(p, index=False)
    return p


def test_identical_numbers_pass_except_the_unvalidated_ambiguous_expiry(tmp_path):
    """Strike/vol/premium equality passes everywhere. NOTE: this is synthetic-vs-our-own-numbers,
    i.e. NOT an external validation; it only checks the comparison machinery."""
    rep = compare_with_bloomberg(_filled_table(tmp_path)).set_index(["tenor", "item"])
    assert len(rep) == 9 * (3 + 10 + 1)
    amb = rep.loc[("1M", "expiry_date (roller)")]
    assert not amb.passed and amb.ours.startswith("ambiguous: 2026-11-24,2026-11-25,2026-11-27")
    assert rep.drop(index=("1M", "expiry_date (roller)")).passed.all()


def test_strike_vol_and_date_gaps_are_flagged(tmp_path):
    def perturb(raw):
        raw.loc[raw.tenor == "1W", "bbg_K_25DP"] += 2e-4                  # 2 pips
        raw.loc[raw.tenor == "1M", "bbg_vol_10DC"] += 0.05                # 0.05 vol pts
        raw.loc[raw.tenor == "2M", "delivery_date"] = "2026-12-29"        # Bloomberg says a different date
    rep = compare_with_bloomberg(_filled_table(tmp_path, perturb)).set_index(["tenor", "item"])
    assert not rep.loc[("1W", "25DP strike"), "passed"]
    assert not rep.loc[("1M", "10DC vol (pts)"), "passed"]
    assert not rep.loc[("2M", "delivery_date (roller)"), "passed"]
    untouched = rep.drop(index=[("1W", "25DP strike"), ("1M", "10DC vol (pts)"), ("1M", "expiry_date (roller)")]).drop(index="2M", level=0)
    assert untouched.passed.all()

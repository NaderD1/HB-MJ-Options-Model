"""Snapshot public proxy option chains (DEVELOPMENT / PROXY data, never research data).

Saves, per ticker, the raw yfinance chain (all expiries up to MAX_DAYS), FRED Treasury yields and
metadata (retrieval time, spot and its timestamp, source) under data/snapshots/, using the same
file layout as the FXE snapshot so load_snapshot() reads them.

Run during US market hours:  python -m scripts.snapshot_proxy FXE SPY
"""

import json
import sys
from pathlib import Path

import pandas as pd

from src.data_loaders import fetch_fred_treasury
from src.timeutils import localize, to_utc

SNAP_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshots"
MAX_DAYS = 120


def fetch_chain(ticker: str) -> tuple[pd.DataFrame, dict]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    retrieved = pd.Timestamp.now(tz="UTC")
    hist = tk.history(period="1d", interval="1m")
    if len(hist):
        spot, spot_ts, kind = float(hist["Close"].iloc[-1]), to_utc(hist.index[-1]), "intraday_1m"
    else:
        daily = tk.history(period="10d")
        spot, spot_ts, kind = float(daily["Close"].iloc[-1]), localize(str(daily.index[-1].date()), "16:00", "America/New_York"), "previous_close"
    frames = []
    for exp in tk.options:
        if (pd.Timestamp(exp).tz_localize("UTC") - retrieved).days > MAX_DAYS:
            break
        ch = tk.option_chain(exp)
        for cp, df in (("C", ch.calls), ("P", ch.puts)):
            df = df.copy()
            df["expiration"], df["cp"] = exp, cp
            frames.append(df)
    meta = {"underlying": ticker, "spot": spot, "spot_ts_utc": spot_ts.isoformat(), "spot_kind": kind,
            "retrieved_utc": retrieved.isoformat(), "source": "yfinance (Yahoo Finance; quotes may be delayed)"}
    return pd.concat(frames, ignore_index=True), meta


def main(tickers):
    rates = fetch_fred_treasury()
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        raw, meta = fetch_chain(t)
        stem = f"{t.lower()}_" + pd.Timestamp(meta["retrieved_utc"]).strftime("%Y%m%dT%H%M%SZ")
        raw.to_csv(SNAP_DIR / f"{stem}_chain.csv", index=False)
        rates.to_csv(SNAP_DIR / f"{stem}_rates.csv", index=False)
        (SNAP_DIR / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{t}: {len(raw)} rows, {raw.expiration.nunique()} expiries, spot {meta['spot']} ({meta['spot_kind']} {meta['spot_ts_utc']}) -> {stem}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["FXE", "SPY"])

"""Daily FXE option-chain snapshot (DEVELOPMENT FALLBACK data, not the research dataset).

Saves the raw yfinance chain, the FRED Treasury yields used for discounting, and metadata
(retrieval time, spot timestamp, sources) under data/snapshots/, so a history can be built
and every normalized row can be traced back to an unmodified raw file.

Run once per day, ideally during US market hours (quotes outside 09:30-16:00 ET are stale):
    python -m scripts.snapshot_fxe
"""

from pathlib import Path

from src.data_loaders import fetch_fred_treasury, fetch_fxe_snapshot, save_snapshot

SNAP_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def main() -> Path:
    raw, meta = fetch_fxe_snapshot()
    rates = fetch_fred_treasury()
    path = save_snapshot(raw, meta, rates, SNAP_DIR)
    print(f"saved {len(raw)} raw option rows -> {path}")
    return path


if __name__ == "__main__":
    main()

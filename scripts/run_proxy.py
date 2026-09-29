"""One command for the proxy (public-data) study from the SAVED snapshots -- no network, no Bloomberg.

    python -m scripts.run_proxy

Runs the calendar-clock study, then the trading-clock study, then the figures. Every model fit is
checkpointed (results/proxy*/fit_results/*.jsonl) and restored on relaunch, so completed fits are
never re-optimised; an interrupted run resumes where it stopped. Data: data/snapshots/
spy_20260929T190930Z_* and fxe_20260929T190930Z_* (committed to the repository).
"""

import runpy
import sys

if __name__ == "__main__":
    for clock in ("calendar", "trading"):
        sys.argv = ["scripts.proxy_study", "--clock", clock]
        runpy.run_module("scripts.proxy_study", run_name="__main__")
    sys.argv = ["scripts.proxy_figures"]
    runpy.run_module("scripts.proxy_figures", run_name="__main__")

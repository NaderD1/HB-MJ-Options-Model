"""Checkpoint / resume logic: append-only rows, corrupt-line tolerance, and restore-without-refit."""

import json

import numpy as np
import pytest

import scripts.proxy_study as ps
from src.calibration import Objective
from src.pooled import PooledSpec, calibrate_pooled_nested, event_parameter_map
from src.pooled_study_utils import true_shared
from tests.test_pooled import ECB, FOMC, _panel


@pytest.fixture()
def tmp_ckpt(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, "CKPT_DIR", tmp_path / "fit_results")
    return tmp_path / "fit_results"


def test_append_and_read_last_row_wins(tmp_ckpt):
    ps.append_checkpoint({"fold_id": "loeo_x", "model": "Bates", "status": "not_converged"})
    ps.append_checkpoint({"fold_id": "loeo_x", "model": "Bates", "status": "converged"})
    ps.append_checkpoint({"fold_id": "full_sample", "model": "Heston", "status": "timed_out"})
    ck = ps.read_checkpoints()
    assert ck[("loeo_x", "Bates")]["status"] == "converged"
    assert ck[("full_sample", "Heston")]["status"] == "timed_out"
    assert (tmp_ckpt / "loeo_x.jsonl").read_text().count("\n") == 2  # append-only: history kept


def test_corrupt_trailing_line_from_killed_writer_is_skipped(tmp_ckpt):
    ps.append_checkpoint({"fold_id": "f", "model": "Heston", "status": "converged"})
    with open(tmp_ckpt / "f.jsonl", "a") as fh:
        fh.write('{"fold_id": "f", "model": "Bat')  # half-written line
    assert set(ps.read_checkpoints()) == {("f", "Heston")}


def test_resume_restores_finished_stages_without_reoptimising():
    panel, truth = _panel((FOMC, ECB))
    mapping, keys, _ = event_parameter_map(panel)
    ts = true_shared(truth, keys)
    done = {}
    for name in ("Heston", "Bates", "HB-MJ"):
        spec = PooledSpec.make(name, keys)
        done[name] = {"shared": {k: ts[k] for k in spec.shared_all}, "local": [{k: d[k] for k in spec.local} for d in truth.daily],
                      "status": 1, "success": True, "message": "saved"}
    calls = []
    nc = calibrate_pooled_nested(panel, done=done, on_stage=lambda n, f: calls.append(n), threads=1)
    assert calls == []                                    # nothing re-fitted, nothing re-checkpointed
    for name, f in nc.best().items():
        assert f.n_eval == 0 and "restored from checkpoint" in f.message
        assert f.shared == done[name]["shared"]
    assert nc.hbmj.iv_rmse < 1e-6                         # truth parameters price the synthetic panel exactly


def test_resume_continues_after_last_finished_stage():
    """Heston saved -> only Bates and HB-MJ run, and each is checkpointed as it finishes."""
    panel, truth = _panel((FOMC, ECB))
    mapping, keys, _ = event_parameter_map(panel)
    ts = true_shared(truth, keys)
    hs = PooledSpec.make("Heston")
    done = {"Heston": {"shared": {k: ts[k] for k in hs.shared_all}, "local": [{k: d[k] for k in hs.local} for d in truth.daily],
                       "status": 1, "success": True, "message": "saved"}}
    calls = []
    calibrate_pooled_nested(panel, done=done, on_stage=lambda n, f: calls.append(n), threads=1, timeout_s=20)
    assert calls == ["Bates", "HB-MJ"]

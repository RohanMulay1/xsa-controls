"""Regression tests for evidence-gate aggregation semantics."""

from __future__ import annotations

from scripts import verify_day
from xsac.config import TRAIN
from xsac.runmeta import RunRecord, write_record


def _grid(size: str, tokens: int):
    seeds = range(8 if size == "s" else 3)
    return [
        {"status": "completed", "arm": arm, "seed": seed,
         "size": size.upper(), "tokens_seen": tokens,
         "final_val_loss": 5.0}
        for seed in seeds for arm in ("baseline", "xsa", "random")
    ]


def test_raw_record_identity_fields_override_metric_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_day, "RESULTS", tmp_path)
    raw = tmp_path / "factorial_s"
    write_record(RunRecord(
        run_id="one", experiment="factorial", status="completed",
        arm="xsa", seed=42, size="S",
        metrics={"status": "failed", "arm": "wrong", "seed": -1,
                 "size": "wrong", "tokens_seen": 349_962_240,
                 "final_val_loss": 5.0}), raw)

    rows, source = verify_day._factorial_rows("s")

    assert source == "factorial_s"
    assert rows == [{"status": "completed", "arm": "xsa", "seed": 42,
                     "size": "S", "tokens_seen": 349_962_240,
                     "final_val_loss": 5.0}]


def test_day46_rejects_uniform_but_subfloor_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_day, "RESULTS", tmp_path)
    floor = int(TRAIN.tokens_min) // TRAIN.batch_tokens * TRAIN.batch_tokens
    budgets = {"s": floor - TRAIN.batch_tokens, "m": floor}
    monkeypatch.setattr(
        verify_day, "_factorial_rows",
        lambda size: (_grid(size, budgets[size]), "synthetic"))

    checks = {name: passed for passed, name, _ in verify_day.day46()}

    assert checks["one token budget across the full grid (s)"] is True
    assert checks["token budget meets the preregistered floor (s)"] is False
    assert checks["token budget meets the preregistered floor (m)"] is True
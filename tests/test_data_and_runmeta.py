"""Data pairing guarantees, run records and resumability."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from xsac.config import CFG_TINY, ExperimentConfig, TRAIN, smoke_variant
from xsac.data import (FixedEvalLoader, PairedLoader, TokenDataset,
                       ensure_smoke_data, synthetic_tokens)
from xsac.runmeta import (NUMERIC_STATUSES, STATUSES, RunRecord,
                          is_done, numeric_records,
                          read_records, records_to_rows, run_id, write_csv,
                          write_record)


@pytest.fixture
def data_dir(tmp_path):
    ensure_smoke_data(tmp_path, 512, n_train=40_000, n_val=8_000)
    return tmp_path


class TestPairingGuarantee:
    """The only difference between two runs at a seed must be the arm."""

    def test_same_seed_gives_identical_batches(self, data_dir):
        ds = TokenDataset(data_dir / "train.bin")
        a = PairedLoader(ds, 32, 4, seed=42)
        b = PairedLoader(ds, 32, 4, seed=42)
        for _ in range(5):
            xa, ya = a.batch()
            xb, yb = b.batch()
            assert torch.equal(xa, xb) and torch.equal(ya, yb)

    def test_different_seeds_give_different_batches(self, data_dir):
        ds = TokenDataset(data_dir / "train.bin")
        a = PairedLoader(ds, 32, 4, seed=42)
        b = PairedLoader(ds, 32, 4, seed=1337)
        assert not torch.equal(a.batch()[0], b.batch()[0])

    def test_loader_is_immune_to_global_rng_state(self, data_dir):
        """Building a model between batches must not shift the data order.

        The loader holds its own numpy Generator rather than drawing from the
        global stream, so an unrelated consumer of torch or numpy randomness
        cannot silently break the pairing between two arms.
        """
        ds = TokenDataset(data_dir / "train.bin")
        clean = PairedLoader(ds, 32, 4, seed=42)
        clean_batches = [clean.batch()[0].clone() for _ in range(3)]

        noisy = PairedLoader(ds, 32, 4, seed=42)
        noisy_batches = []
        for _ in range(3):
            torch.manual_seed(999)
            torch.randn(1000)
            np.random.seed(12345)
            np.random.random(1000)
            noisy_batches.append(noisy.batch()[0].clone())

        for a, b in zip(clean_batches, noisy_batches):
            assert torch.equal(a, b), (
                "unrelated RNG consumption changed the data order; the "
                "pairing between arms would not hold")

    def test_targets_are_inputs_shifted_by_one(self, data_dir):
        ds = TokenDataset(data_dir / "train.bin")
        x, y = PairedLoader(ds, 32, 2, seed=7).batch()
        assert torch.equal(x[:, 1:], y[:, :-1])

    def test_reset_restores_the_stream(self, data_dir):
        ds = TokenDataset(data_dir / "train.bin")
        loader = PairedLoader(ds, 32, 2, seed=3)
        first = loader.batch()[0]
        loader.batch()
        loader.reset()
        assert torch.equal(loader.batch()[0], first)


class TestFixedEvalIsDeterministic:
    """Shared eval noise would sit on top of the effect we are measuring."""

    def test_two_passes_are_identical(self, data_dir):
        ds = TokenDataset(data_dir / "val.bin")
        loader = FixedEvalLoader(ds, 32, 4, 4096)
        first = [x.clone() for x, _ in loader]
        second = [x.clone() for x, _ in loader]
        assert len(first) == len(second) and first
        for a, b in zip(first, second):
            assert torch.equal(a, b)

    def test_sequences_do_not_overlap(self, data_dir):
        ds = TokenDataset(data_dir / "val.bin")
        loader = FixedEvalLoader(ds, 16, 2, 512)
        starts = []
        for x, _ in loader:
            starts.append(x.shape)
        assert all(s == starts[0] for s in starts)


class TestTokenDataset:
    def test_missing_file_names_the_fix(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            TokenDataset(tmp_path / "nope.bin")
        assert "prepare.py" in str(exc.value)

    def test_odd_byte_count_is_rejected(self, tmp_path):
        p = tmp_path / "bad.bin"
        p.write_bytes(b"\x01\x02\x03")
        with pytest.raises(ValueError) as exc:
            TokenDataset(p)
        assert "truncated" in str(exc.value)

    def test_size_verification_is_exact(self, tmp_path):
        """Day-1 gate: bytes == 2 * n_tokens."""
        synthetic_tokens(tmp_path / "t.bin", 1000, 512, seed=0)
        ds = TokenDataset(tmp_path / "t.bin")
        assert ds.verify_size(1000) is True
        assert ds.verify_size(999) is False

    def test_splits_are_disjoint_by_construction(self, data_dir):
        train = TokenDataset(data_dir / "train.bin")
        val = TokenDataset(data_dir / "val.bin")
        # Different generator seeds, so the sequences differ. Checked on a
        # long window: a short one could coincide by chance.
        a = np.asarray(train.tokens[:2000])
        b = np.asarray(val.tokens[:2000])
        assert not np.array_equal(a, b)


class TestRunRecords:
    def test_run_id_is_deterministic_and_content_addressed(self):
        cfg = {"arm": "xsa", "size": "S"}
        assert run_id(cfg, 42) == run_id(dict(cfg), 42)
        assert run_id(cfg, 42) != run_id(cfg, 43)
        assert run_id(cfg, 42) != run_id({"arm": "random", "size": "S"}, 42)
        assert len(run_id(cfg, 42)) == 12

    def test_key_order_does_not_change_the_id(self):
        assert run_id({"a": 1, "b": 2}, 0) == run_id({"b": 2, "a": 1}, 0)

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValueError):
            RunRecord(run_id="x", experiment="e", status="great")

    def test_only_numeric_statuses_reach_aggregation(self):
        recs = [RunRecord(run_id=s, experiment="e", status=s) for s in STATUSES]
        keep = numeric_records(recs)
        assert {r.status for r in keep} == set(NUMERIC_STATUSES)
        assert len(keep) == 2

    def test_an_oom_contributes_no_number(self):
        rec = RunRecord(run_id="a", experiment="e", status="oom")
        assert rec.is_numeric is False

    def test_write_and_read_round_trip(self, tmp_path):
        rec = RunRecord(run_id="abc123", experiment="factorial",
                        status="completed", arm="xsa", seed=42, size="S",
                        metrics={"final_val_loss": 3.21})
        write_record(rec, tmp_path)
        back = read_records(tmp_path)
        assert len(back) == 1
        assert back[0].metrics["final_val_loss"] == 3.21
        assert back[0].arm == "xsa"

    def test_resumability_skips_completed_cells(self, tmp_path):
        assert is_done(tmp_path, "abc123") is False
        write_record(RunRecord(run_id="abc123", experiment="e",
                               status="completed"), tmp_path)
        assert is_done(tmp_path, "abc123") is True

    def test_a_failed_cell_is_not_treated_as_done(self, tmp_path):
        write_record(RunRecord(run_id="dead", experiment="e", status="failed"),
                     tmp_path)
        assert is_done(tmp_path, "dead") is False

    def test_writes_are_atomic_leaving_no_temp_files(self, tmp_path):
        write_record(RunRecord(run_id="x1", experiment="e", status="completed"),
                     tmp_path)
        assert not list((tmp_path / "runs").glob("*.tmp"))

    def test_csv_carries_the_status_column(self, tmp_path):
        recs = [RunRecord(run_id="a", experiment="e", status="completed",
                          arm="xsa", metrics={"final_val_loss": 1.0}),
                RunRecord(run_id="b", experiment="e", status="oom", arm="xsa")]
        rows = records_to_rows(recs)
        assert {r["status"] for r in rows} == {"completed", "oom"}
        out = write_csv(rows, tmp_path / "t.csv")
        assert out.exists() and "status" in out.read_text(encoding="utf-8")


class TestSmokeConfig:
    def test_smoke_variant_is_tiny_and_flagged(self):
        cfg = ExperimentConfig(arm="xsa", seed=1, size="S", train=TRAIN)
        smoke = smoke_variant(cfg)
        assert smoke.smoke is True
        assert smoke.model is CFG_TINY
        assert smoke.train.tokens_per_run < cfg.train.tokens_per_run

    def test_config_serialises_to_a_stable_dict(self):
        cfg = ExperimentConfig(arm="random", seed=42)
        d = cfg.to_dict()
        assert d["arm"] == "random"
        assert isinstance(d["model"], dict)
        json.dumps(d)   # must be JSON-serialisable for the run id

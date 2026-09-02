"""Track A measurement, figures, and end-to-end integration.

Tests needing a network download are marked ``slow`` and excluded from the
default CI job.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from xsac.frozen import (aggregate_model, expand_kv, gqa_within_across,
                         group_of_head, self_and_null_cosines)


class TestGQAExpansion:
    """It must match HF ``repeat_kv`` exactly, or every GQA number is wrong."""

    @pytest.mark.parametrize("n_rep", [1, 2, 4, 8])
    def test_matches_huggingface_repeat_kv(self, n_rep):
        from transformers.models.llama.modeling_llama import repeat_kv
        v = torch.randn(2, 4, 7, 8)
        assert torch.equal(expand_kv(v, 4 * n_rep), repeat_kv(v, n_rep))

    def test_mha_is_the_identity(self):
        v = torch.randn(1, 6, 5, 4)
        assert expand_kv(v, 6) is v

    def test_non_divisible_head_counts_raise(self):
        with pytest.raises(ValueError):
            expand_kv(torch.randn(1, 3, 4, 2), 8)

    def test_expansion_is_contiguous_per_group(self):
        """Query head h must use KV head h // n_rep, not an interleaving."""
        v = torch.zeros(1, 2, 1, 3)
        v[0, 0] = 1.0
        v[0, 1] = 2.0
        out = expand_kv(v, 6)
        assert [float(out[0, h, 0, 0]) for h in range(6)] == [1, 1, 1, 2, 2, 2]

    def test_group_of_head_agrees_with_the_expansion(self):
        assert [group_of_head(h, 6, 2) for h in range(6)] == [0, 0, 0, 1, 1, 1]

    def test_group_of_head_rejects_bad_counts(self):
        with pytest.raises(ValueError):
            group_of_head(0, 6, 4)


class TestCosineMeasurement:
    def test_identical_vectors_give_cosine_one(self):
        v = torch.randn(1, 2, 8, 4)
        cs, _ = self_and_null_cosines(v.clone(), v)
        assert torch.allclose(cs, torch.ones_like(cs), atol=1e-5)

    def test_position_zero_is_excluded_by_default(self):
        """cos(y_0, v_0) is 1 by construction and carries no information."""
        y = torch.randn(1, 2, 10, 4)
        cs, _ = self_and_null_cosines(y, y, min_position=1)
        assert cs.shape[-1] == 9

    def test_including_position_zero_changes_the_count(self):
        y = torch.randn(1, 2, 10, 4)
        cs, _ = self_and_null_cosines(y, y, min_position=0)
        assert cs.shape[-1] == 10

    def test_null_partner_is_a_legal_causal_key(self):
        """j must come from [0, i): a position the query could have attended."""
        torch.manual_seed(0)
        y = torch.randn(1, 1, 32, 4)
        v = torch.randn(1, 1, 32, 4)
        # Make v position-identifiable so we can recover which j was drawn.
        for t in range(32):
            v[0, 0, t] = float(t)
        _, cn = self_and_null_cosines(y, v, min_position=1)
        assert torch.isfinite(cn).all()

    def test_too_short_a_sequence_returns_empty_rather_than_crashing(self):
        y = torch.randn(1, 2, 1, 4)
        cs, cn = self_and_null_cosines(y, y)
        assert cs.numel() == 0 and cn.numel() == 0

    def test_zero_vectors_do_not_produce_nan(self):
        y = torch.zeros(1, 2, 8, 4)
        v = torch.randn(1, 2, 8, 4)
        cs, cn = self_and_null_cosines(y, v)
        assert torch.isfinite(cs).all() and torch.isfinite(cn).all()


class TestAggregation:
    def test_empty_input_is_nan_not_zero(self):
        assert math.isnan(aggregate_model([])["cos_self"])

    def test_self_specific_fraction(self):
        rows = [{"cos_self": 0.5406, "cos_null": 0.3798, "excess": 0.1608}]
        agg = aggregate_model(rows)
        assert agg["self_specific_fraction"] == pytest.approx(0.297, abs=0.002)

    def test_mha_split_is_reported_as_non_existent(self):
        """GPT-2 is 12 query / 12 KV heads. It must not read as grouped."""
        rows = [{"cos_self": 0.5, "cos_null": 0.3, "excess": 0.2,
                 "cos_across_group": float("nan"), "layer": l, "head": h,
                 "kv_group": h, "n_kv_heads": 12}
                for l in range(12) for h in range(12)]
        out = gqa_within_across(rows)
        assert out["is_gqa"] is False
        assert out["n_query_heads"] == 12
        assert "MHA" in out["note"]
        assert math.isnan(out["within_group_excess"])
        assert math.isnan(out["across_group_excess"])

    def test_groups_do_not_span_layers(self):
        """Keying by kv_group alone bucketed head 3 of every layer together."""
        rows = [{"cos_self": 0.5, "cos_null": 0.3, "excess": 0.2,
                 "cos_across_group": float("nan"), "layer": l, "head": h,
                 "kv_group": h, "n_kv_heads": 4}
                for l in range(5) for h in range(4)]
        assert gqa_within_across(rows)["n_groups"] == 20

    def test_gqa_split_reports_distinct_within_and_across(self):
        rows = [{"cos_self": 0.5, "cos_null": 0.3, "cos_across_group": 0.35,
                 "layer": l, "head": h, "kv_group": h // 4, "n_kv_heads": 2}
                for l in range(3) for h in range(8)]
        out = gqa_within_across(rows)
        assert out["is_gqa"] is True
        assert out["n_query_heads"] == 8 and out["heads_per_kv"] == 4
        assert out["n_groups"] == 6
        assert out["within_group_excess"] == pytest.approx(0.2)
        assert out["across_group_excess"] == pytest.approx(0.05)


@pytest.mark.slow
class TestAgainstRealModels:
    """Requires a network download. Excluded from the fast CI job."""

    @pytest.mark.parametrize("model_id", [
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "sshleifer/tiny-gpt2",
        "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    ])
    def test_probe_runs_across_architectures(self, model_id):
        from xsac.frozen import FrozenProbe
        probe = FrozenProbe.from_pretrained(model_id)
        rows = probe.measure([torch.randint(0, 50, (1, 24))], layers=[0])
        assert rows, "no rows produced for {}".format(model_id)
        for r in rows:
            assert math.isfinite(r["cos_self"])
            assert math.isfinite(r["cos_null"])
            assert r["excess"] == pytest.approx(r["cos_self"] - r["cos_null"])

    def test_eager_attention_is_forced(self):
        """SDPA does not expose the attention matrix; a silent fallback would
        compute the null from the wrong tensor."""
        from xsac.frozen import FrozenProbe
        probe = FrozenProbe.from_pretrained("sshleifer/tiny-gpt2")
        impl = getattr(probe.model.config, "_attn_implementation", "eager")
        assert impl == "eager"


class TestFigures:
    def test_a_figure_with_no_data_is_skipped_not_drawn(self, tmp_path):
        from xsac.figures import FigureSkipped, fig3_ladder
        with pytest.raises(FigureSkipped):
            fig3_ladder(tmp_path, tmp_path / "out")
        assert not (tmp_path / "out").exists() or not list(
            (tmp_path / "out").glob("*.png"))

    def test_every_figure_writes_png_pdf_and_data(self, tmp_path):
        from xsac.figures import fig5_gqa
        import csv
        src = tmp_path / "gqa.csv"
        with src.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "is_gqa",
                                               "within_group_excess",
                                               "across_group_excess"])
            w.writeheader()
            w.writerow({"model": "m1", "is_gqa": "True",
                        "within_group_excess": 0.2,
                        "across_group_excess": 0.15})
        out = tmp_path / "figs"
        paths = fig5_gqa(tmp_path, out)
        names = {p.suffix for p in paths}
        assert names == {".png", ".pdf", ".csv"}
        assert (out / "fig5_gqa_data.csv").exists()

    def test_unknown_model_in_ladder_raises_rather_than_guessing_size(
            self, tmp_path):
        from xsac.figures import FigureSkipped, fig3_ladder
        import csv
        src = tmp_path / "ladder.csv"
        with src.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "cos_self",
                                               "cos_null", "excess"])
            w.writeheader()
            w.writerow({"model": "not-a-real-model", "cos_self": 0.5,
                        "cos_null": 0.3, "excess": 0.2})
        with pytest.raises(FigureSkipped) as exc:
            fig3_ladder(tmp_path, tmp_path / "out")
        assert "rather than guessing" in str(exc.value)


class TestEndToEnd:
    def test_a_smoke_factorial_cell_trains_and_records(self, tmp_path):
        from xsac.config import ExperimentConfig, TRAIN, smoke_variant
        from xsac.data import ensure_smoke_data
        from xsac.train import train_one

        cfg = smoke_variant(ExperimentConfig(arm="random", seed=42, size="S",
                                             train=TRAIN))
        ensure_smoke_data(tmp_path, cfg.model.vocab_size,
                          n_train=40_000, n_val=8_000)
        out = train_one(cfg, tmp_path, device="cpu", max_steps=2)
        assert math.isfinite(out["final_val_loss"])
        assert out["tokens_seen"] > 0
        assert out["learned_alpha"]

    def test_all_arms_see_identical_tokens_at_one_seed(self, tmp_path):
        """The Days 4-6 gate's proof that the pairing held."""
        from xsac.config import ExperimentConfig, TRAIN, smoke_variant
        from xsac.data import ensure_smoke_data
        from xsac.train import train_one

        ensure_smoke_data(tmp_path, 512, n_train=40_000, n_val=8_000)
        seen = set()
        for arm in ("baseline", "xsa", "random", "meanval", "diagmask"):
            cfg = smoke_variant(ExperimentConfig(arm=arm, seed=7, size="S",
                                                 train=TRAIN))
            out = train_one(cfg, tmp_path, device="cpu", max_steps=2)
            seen.add(out["tokens_seen"])
        assert len(seen) == 1, "arms saw different token counts: {}".format(seen)

    def test_step0_losses_are_equal_across_arms(self, tmp_path):
        from xsac.config import ARMS, CFG_TINY
        from xsac.data import ensure_smoke_data
        from xsac.train import step0_losses

        ensure_smoke_data(tmp_path, CFG_TINY.vocab_size,
                          n_train=40_000, n_val=8_000)
        losses = step0_losses(list(ARMS), CFG_TINY, 42, tmp_path)
        base = losses["baseline"]
        assert max(abs(v - base) for v in losses.values()) < 1e-6

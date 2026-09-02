"""Model correctness: parameter counts, pairing, diagmask, position 0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from xsac.config import ARMS, CFG_M, CFG_S, CFG_TINY
from xsac.model import DIAG_MASK_STRENGTH, GPT, apply_rope, build_rope_cache


class TestParameterCounts:
    """RoPE is not stated in the spec; the parameter counts imply it.

    With RoPE, CFG_S is 50.93M total and 25.2M non-embedding, matching the
    spec's "~50.9M / ~25.2M" exactly. Learned position embeddings would add
    1024 * n_embd and give 51.45M, which misses. These tests pin the choice so
    it cannot drift back.
    """

    @pytest.mark.parametrize("cfg,total_m,non_emb_m", [
        (CFG_S, 50.9, 25.2),
        (CFG_M, 123.6, 84.9),
    ])
    def test_analytic_matches_the_spec(self, cfg, total_m, non_emb_m):
        total, non_emb = cfg.analytic_params()
        assert total / 1e6 == pytest.approx(total_m, abs=0.15)
        assert non_emb / 1e6 == pytest.approx(non_emb_m, abs=0.15)

    def test_built_model_matches_its_own_analytic_count(self):
        model = GPT(CFG_TINY, arm="baseline")
        total, non_emb = CFG_TINY.analytic_params()
        assert model.num_params() == total
        assert model.num_params(non_embedding=True) == non_emb

    def test_no_learned_position_embedding_exists(self):
        model = GPT(CFG_TINY)
        names = dict(model.named_parameters())
        assert not any("pos" in n or "wpe" in n for n in names), (
            "a learned position embedding would break the parameter counts "
            "and change them with context length")

    def test_weights_are_tied(self):
        model = GPT(CFG_TINY)
        assert model.lm_head.weight is model.wte.weight


class TestRoPE:
    def test_cache_shape_and_range(self):
        cos, sin = build_rope_cache(32, 8, 10000.0, torch.device("cpu"))
        assert cos.shape == (1, 1, 32, 8)
        assert float(cos.abs().max()) <= 1.0 + 1e-6

    def test_rotation_preserves_norm(self):
        cos, sin = build_rope_cache(16, 8, 10000.0, torch.device("cpu"))
        x = torch.randn(2, 4, 16, 8)
        assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1),
                              x.norm(dim=-1), atol=1e-5)

    def test_odd_head_dim_is_rejected(self):
        with pytest.raises(ValueError):
            build_rope_cache(8, 7, 10000.0, torch.device("cpu"))


class TestCrossArmPairing:
    """Self-test 10. If this fails the statistical design is void."""

    def test_every_arm_matches_baseline_at_step_zero(self):
        g = torch.Generator().manual_seed(5)
        x = torch.randint(0, CFG_TINY.vocab_size, (2, 24), generator=g)
        y = torch.randint(0, CFG_TINY.vocab_size, (2, 24), generator=g)
        losses = {}
        for arm in ARMS:
            torch.manual_seed(77)
            m = GPT(CFG_TINY, arm=arm)
            m.eval()
            with torch.no_grad():
                _, loss = m(x, y)
            losses[arm] = float(loss)
        base = losses["baseline"]
        worst = max(abs(v - base) for v in losses.values())
        assert worst < 1e-6, "arms diverge at step 0: {}".format(losses)

    def test_shared_parameters_are_identical_across_arms(self):
        torch.manual_seed(3)
        a = GPT(CFG_TINY, arm="baseline")
        torch.manual_seed(3)
        b = GPT(CFG_TINY, arm="random")
        for name, pa in a.named_parameters():
            pb = dict(b.named_parameters())[name]
            assert torch.equal(pa, pb), "{} differs across arms".format(name)


class TestDiagMask:
    def test_position_zero_row_is_intact_on_both_paths(self):
        idx = torch.randint(0, CFG_TINY.vocab_size, (2, 18))
        for hard in (False, True):
            torch.manual_seed(0)
            m = GPT(CFG_TINY, arm="diagmask", diagmask_hard=hard)
            m.eval()
            caps = [dict() for _ in range(CFG_TINY.n_layer)]
            with torch.no_grad():
                logits, _ = m(idx, captures=caps)
            att = caps[0]["att"]
            assert torch.isfinite(att).all(), "NaN with hard={}".format(hard)
            assert torch.isfinite(logits).all()
            assert float(att[..., 0, 0].min()) == pytest.approx(1.0, abs=1e-6)

    def test_gated_path_suppresses_the_diagonal_at_full_strength(self):
        torch.manual_seed(0)
        m = GPT(CFG_TINY, arm="diagmask")
        for b in m.h:
            with torch.no_grad():
                b.attn.diag_alpha.fill_(10.0)
        m.eval()
        caps = [dict() for _ in range(CFG_TINY.n_layer)]
        with torch.no_grad():
            m(torch.randint(0, CFG_TINY.vocab_size, (1, 12)), captures=caps)
        att = caps[0]["att"]
        diag = att[..., torch.arange(1, 12), torch.arange(1, 12)]
        assert float(diag.abs().max()) < 1e-9
        rows = att.sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)

    def test_hard_path_deviates_from_baseline_at_step_zero(self):
        """The measured gap that motivates the gated default.

        The spec requires all five arms within 1e-6 of baseline at step 0, and
        also shows a hard -inf snippet. Both cannot hold. This records the
        size of the conflict instead of hiding it.
        """
        g = torch.Generator().manual_seed(5)
        x = torch.randint(0, CFG_TINY.vocab_size, (2, 24), generator=g)
        y = torch.randint(0, CFG_TINY.vocab_size, (2, 24), generator=g)

        def loss_for(arm, hard=False):
            torch.manual_seed(77)
            m = GPT(CFG_TINY, arm=arm, diagmask_hard=hard)
            m.eval()
            with torch.no_grad():
                _, l = m(x, y)
            return float(l)

        gap = abs(loss_for("diagmask", hard=True) - loss_for("baseline"))
        assert gap > 1e-6, (
            "the hard mask is expected to deviate at step 0; if it no longer "
            "does, the gated default is unnecessary")

    def test_gated_alpha_is_in_the_optimiser_and_gate_table(self):
        m = GPT(CFG_TINY, arm="diagmask")
        assert len(m.alpha_parameters()) == CFG_TINY.n_layer
        table = m.gate_table()
        assert len(table) == CFG_TINY.n_layer
        assert all(len(v) == CFG_TINY.n_head for v in table.values())

    def test_mask_strength_is_effectively_negative_infinity(self):
        import math
        assert math.exp(-DIAG_MASK_STRENGTH) < 1e-12


class TestPositionZeroDegeneracy:
    def test_y0_equals_v0_exactly(self):
        torch.manual_seed(0)
        m = GPT(CFG_TINY, arm="baseline")
        m.eval()
        caps = [dict() for _ in range(CFG_TINY.n_layer)]
        with torch.no_grad():
            m(torch.randint(0, CFG_TINY.vocab_size, (1, 20)), captures=caps)
        for cap in caps:
            assert torch.allclose(cap["y"][0, :, 0, :], cap["v"][0, :, 0, :],
                                  atol=1e-6)

    def test_xsa_output_at_position_zero_is_zero(self):
        torch.manual_seed(0)
        m = GPT(CFG_TINY, arm="xsa")
        for b in m.h:
            with torch.no_grad():
                b.attn.arm.alpha.fill_(10.0)
        m.eval()
        caps = [dict() for _ in range(CFG_TINY.n_layer)]
        with torch.no_grad():
            m(torch.randint(0, CFG_TINY.vocab_size, (1, 20)), captures=caps)
        assert float(caps[0]["y"][0, :, 0, :].norm(dim=-1).max()) < 1e-5


class TestAttentionPaths:
    def test_explicit_attention_matches_sdpa_for_baseline(self):
        """Capture must not change the answer it is capturing."""
        torch.manual_seed(0)
        m = GPT(CFG_TINY, arm="baseline")
        m.eval()
        idx = torch.randint(0, CFG_TINY.vocab_size, (2, 20))
        with torch.no_grad():
            fast, _ = m(idx)
            caps = [dict() for _ in range(CFG_TINY.n_layer)]
            slow, _ = m(idx, captures=caps)
        assert torch.allclose(fast, slow, atol=1e-5)

    def test_attention_is_causal(self):
        torch.manual_seed(0)
        m = GPT(CFG_TINY, arm="baseline")
        m.eval()
        caps = [dict() for _ in range(CFG_TINY.n_layer)]
        with torch.no_grad():
            m(torch.randint(0, CFG_TINY.vocab_size, (1, 16)), captures=caps)
        att = caps[0]["att"][0, 0]
        upper = torch.triu(att, diagonal=1)
        assert float(upper.abs().max()) == 0.0

    def test_sequence_longer_than_block_size_is_rejected(self):
        m = GPT(CFG_TINY)
        with pytest.raises(ValueError):
            m(torch.zeros(1, CFG_TINY.block_size + 1, dtype=torch.long))

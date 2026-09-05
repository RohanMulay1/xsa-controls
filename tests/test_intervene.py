"""Tests for the frozen-model XSA intervention.

The reconstruction gate is the load-bearing one. If ``A @ expand_kv(V)`` does
not reproduce what the model feeds to its output projection, then the head
layout or the KV expansion is wrong and every per-head number downstream is
measuring the wrong object -- while still looking like a number. Everything
else here guards against the intervention silently doing nothing, which would
produce a delta of exactly zero and read as a null result.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from xsac.arms import XSA  # noqa: E402
from xsac.intervene import (InterventionNotApplied,  # noqa: E402
                            ReconstructionError, head_loss_deltas, mean_loss,
                            out_projection, remove_self_value,
                            verify_reconstruction, xsa_intervention)

TINY_NEOX = "hf-internal-testing/tiny-random-GPTNeoXForCausalLM"
TINY_GPT2 = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def neox():
    from xsac.frozen import FrozenProbe
    return FrozenProbe.from_pretrained(TINY_NEOX, device="cpu")


@pytest.fixture(scope="module")
def ids():
    torch.manual_seed(0)
    return torch.randint(0, 100, (2, 16))


class TestRemovalMatchesTheTrainedArm:
    def test_frozen_removal_equals_arms_xsa_with_the_gate_open(self):
        """The frozen path and the trained path must be the same operation.
        Two implementations of one intervention is how they drift apart."""
        torch.manual_seed(0)
        y = torch.randn(2, 5, 3, 8)
        v = torch.randn(2, 5, 3, 8)
        ours = remove_self_value(y, v, strength=1.0)

        arm = XSA(n_head=3)
        with torch.no_grad():
            arm.alpha.fill_(10.0)          # tanh(10) ~ 1, gate fully open
        # arms.XSA works in (B, H, T, D); the frozen path in (B, T, H, D).
        theirs = arm(y.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
        theirs = theirs.permute(0, 2, 1, 3)
        assert torch.allclose(ours, theirs, atol=1e-5)

    def test_removal_leaves_no_component_along_v(self):
        torch.manual_seed(0)
        y, v = torch.randn(4, 8), torch.randn(4, 8)
        z = remove_self_value(y, v)
        vhat = v / v.norm(dim=-1, keepdim=True)
        assert torch.allclose((z * vhat).sum(-1),
                              torch.zeros(4), atol=1e-6)

    def test_zero_strength_is_the_identity(self):
        torch.manual_seed(0)
        y, v = torch.randn(4, 8), torch.randn(4, 8)
        assert torch.allclose(remove_self_value(y, v, 0.0), y)


class TestReconstructionGate:
    def test_gate_passes_on_gptneox(self, neox, ids):
        m = verify_reconstruction(neox, ids, 0)
        assert m["max_rel_error"] < 1e-4

    def test_gate_passes_on_gpt2_conv1d(self, ids):
        """GPT-2 uses Conv1D with a fused c_attn, a different layout from
        GPT-NeoX. The gate has to hold for both or the ladder mixes them."""
        from xsac.frozen import FrozenProbe
        probe = FrozenProbe.from_pretrained(TINY_GPT2, device="cpu")
        m = verify_reconstruction(probe, ids, 0)
        assert m["max_rel_error"] < 1e-4

    def test_nan_fails_closed(self, neox, ids, monkeypatch):
        """`nan > tol` is False, so the natural form of the comparison lets a
        reconstruction that produced no usable numbers through as a pass."""
        import xsac.intervene as mod
        real = neox._values_from_hidden

        def poisoned(layer_idx, x):
            v = real(layer_idx, x)
            if v is None:
                return None
            v = v.clone()
            v[..., 0] = float("nan")
            return v

        monkeypatch.setattr(neox, "_values_from_hidden", poisoned)
        with pytest.raises(ReconstructionError):
            mod.verify_reconstruction(neox, ids, 0)

    def test_a_wrong_expansion_is_caught(self, neox, ids, monkeypatch):
        """Corrupt the value vectors and confirm the gate fails. A gate that
        cannot fail is not a gate."""
        import xsac.intervene as mod
        real = neox._values_from_hidden

        def scrambled(layer_idx, x):
            v = real(layer_idx, x)
            return None if v is None else v.flip(-1)

        monkeypatch.setattr(neox, "_values_from_hidden", scrambled)
        with pytest.raises(ReconstructionError, match="does not reproduce"):
            mod.verify_reconstruction(neox, ids, 0)


class TestInterventionActuallyApplies:
    def test_hook_fires_and_moves_the_tensor(self, neox, ids):
        with xsa_intervention(neox, 0, [0]) as hook:
            mean_loss(neox, [ids])
        assert hook.n_applied > 0
        assert hook.total_change > 0.0

    def test_loss_changes_under_intervention(self, neox, ids):
        base = mean_loss(neox, [ids])
        with xsa_intervention(neox, 0, None):
            after = mean_loss(neox, [ids])
        assert after != base

    def test_deltas_are_per_head_and_differ(self, neox, ids):
        nq, _ = neox.head_counts()
        deltas = head_loss_deltas(neox, [ids], 0, range(nq))
        assert len(deltas) == nq
        assert len(set(round(v, 9) for v in deltas.values())) > 1

    def test_a_hook_that_cannot_apply_raises_rather_than_reporting_zero(
            self, neox, ids, monkeypatch):
        """The failure this guards against is silent: no values means no
        removal, delta 0, and a null result that was never measured."""
        monkeypatch.setattr(neox, "_values_from_hidden",
                            lambda layer_idx, x: None)
        with pytest.raises(InterventionNotApplied):
            with xsa_intervention(neox, 0, [0]):
                mean_loss(neox, [ids])

    def test_intervention_is_removed_on_exit(self, neox, ids):
        base = mean_loss(neox, [ids])
        with xsa_intervention(neox, 0, None):
            pass
        assert mean_loss(neox, [ids]) == pytest.approx(base)


class TestOutputProjectionDiscovery:
    def test_finds_the_projection_on_both_architectures(self, neox):
        attn = neox._attn_module(neox._layers()[0])
        _, name = out_projection(attn)
        assert name == "dense"

    def test_unknown_architecture_is_an_error_not_a_guess(self):
        class Bare(torch.nn.Module):
            pass
        with pytest.raises(AttributeError, match="no output projection"):
            out_projection(Bare())


class TestPositionZeroAlignment:
    """The diagnostic excludes position 0; the intervention must too.

    At position 0 causal softmax runs over one element, so a_00 = 1 and
    y_0 = v_0 exactly. Removing the self-value component there zeroes the
    whole head output. Measuring the statistic without position 0 while
    intervening on it compares two different objects.
    """

    def test_position_zero_is_left_alone_by_default(self, neox, ids):
        """Call the rewrite directly: the registered hook is bound at enter,
        so patching the method afterwards would not intercept anything."""
        from xsac.intervene import HeadIntervention, out_projection
        attn = neox._attn_module(neox._layers()[0])
        proj, _ = out_projection(attn)

        grabbed = {}

        def keep_hidden(module, args, kwargs):
            if args:
                grabbed["hidden"] = args[0].detach()
            elif "hidden_states" in kwargs:
                grabbed["hidden"] = kwargs["hidden_states"].detach()

        def keep_y(module, args):
            grabbed["y"] = args[0].detach()

        h1 = attn.register_forward_pre_hook(keep_hidden, with_kwargs=True)
        h2 = proj.register_forward_pre_hook(keep_y)
        try:
            with torch.no_grad():
                neox.model(ids, use_cache=False)
        finally:
            h1.remove()
            h2.remove()

        hook = HeadIntervention(neox, 0, [0], min_position=1)
        hook._hidden = grabbed["hidden"]
        y = grabbed["y"]
        out = hook._rewrite(proj, (y,))[0]

        nq, _ = neox.head_counts()
        hd = y.shape[-1] // nq
        before = y.view(y.shape[0], y.shape[1], nq, hd)
        after = out.view(out.shape[0], out.shape[1], nq, hd)
        assert torch.allclose(before[:, 0, 0, :], after[:, 0, 0, :]),             "position 0 was modified despite min_position=1"
        assert not torch.allclose(before[:, 1:, 0, :], after[:, 1:, 0, :]),             "positions >= 1 were not modified"
        # Other heads untouched.
        assert torch.allclose(before[:, :, 1, :], after[:, :, 1, :])

    def test_min_position_zero_restores_the_old_behaviour(self, neox, ids):
        base = mean_loss(neox, [ids])
        with xsa_intervention(neox, 0, [0], min_position=1):
            guarded = mean_loss(neox, [ids])
        with xsa_intervention(neox, 0, [0], min_position=0):
            unguarded = mean_loss(neox, [ids])
        assert guarded != base and unguarded != base
        assert guarded != unguarded

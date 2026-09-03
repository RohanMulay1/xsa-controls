"""Arm semantics. These mirror the ten self-tests as importable pytest cases.

The self-test script is the Day-1 gate and prints a report; these are the same
invariants in a form CI can run on every push.
"""

from __future__ import annotations

import pytest
import torch

from xsac.arms import (EPS, Baseline, MeanValue, RandomDir, XSA, LOGIT_ARMS,
                       OUTPUT_ARMS, build_arm)

H, DH = 4, 8


def yv(seed=0, b=2, t=16):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, H, t, DH, generator=g),
            torch.randn(b, H, t, DH, generator=g))


class TestZeroInitIsExactlyBaseline:
    """The property the entire paired design rests on."""

    @pytest.mark.parametrize("name", ["baseline", "xsa", "random", "meanval"])
    def test_arm_returns_y_unchanged_at_alpha_zero(self, name):
        y, v = yv(1)
        arm = build_arm(name, H, DH, 0)
        arm.eval()
        assert torch.equal(arm(y, v), y), (
            "{} is not the identity at alpha=0; every paired difference would "
            "confound the intervention with a different starting point".format(
                name))

    def test_gates_start_at_zero(self):
        for name in ("xsa", "random", "meanval"):
            arm = build_arm(name, H, DH, 0)
            assert torch.equal(arm.alpha, torch.zeros(H))
            assert torch.equal(arm.gate_values(), torch.zeros(H))


class TestFullStrengthRemovesTheDirection:
    def test_xsa_output_is_orthogonal_to_v(self):
        y, v = yv(2)
        arm = XSA(H)
        with torch.no_grad():
            arm.alpha.fill_(10.0)
        z = arm(y, v)
        vhat = v / v.norm(dim=-1, keepdim=True)
        assert float((z * vhat).sum(-1).abs().max()) < 1e-4

    def test_randomdir_output_is_orthogonal_to_r(self):
        y, v = yv(3)
        arm = RandomDir(H, DH, 0)
        with torch.no_grad():
            arm.alpha.fill_(10.0)
        z = arm(y, v)
        rhat = arm.r / arm.r.norm(dim=-1, keepdim=True)
        assert float((z * rhat).sum(-1).abs().max()) < 1e-4

    def test_a_zero_value_vector_does_not_produce_nan(self):
        """XSA's published algorithm has no epsilon on ||v||^2. We add one.

        Without it a zero value vector divides by zero and the NaN propagates
        silently through the rest of training.
        """
        y, v = yv(4)
        v[0, 0, 0, :] = 0.0
        arm = XSA(H)
        with torch.no_grad():
            arm.alpha.fill_(1.0)
        out = arm(y, v)
        assert torch.isfinite(out).all()
        assert EPS > 0


class TestRandomDirIsAFixedArbitraryDirection:
    """Not a resampled one. Those are different hypotheses."""

    def test_direction_ignores_the_run_seed(self):
        torch.manual_seed(1)
        a = RandomDir(H, DH, 5)
        torch.manual_seed(999999)
        b = RandomDir(H, DH, 5)
        assert torch.equal(a.r, b.r)

    def test_direction_varies_across_layers(self):
        a, b = RandomDir(H, DH, 1), RandomDir(H, DH, 2)
        assert not torch.equal(a.r, b.r)

    def test_direction_is_unit_norm(self):
        arm = RandomDir(H, DH, 0)
        norms = arm.r.squeeze().norm(dim=-1)
        assert torch.allclose(norms, torch.ones(H), atol=1e-6)

    def test_optimiser_does_not_move_it(self):
        y, v = yv(5)
        arm = RandomDir(H, DH, 0)
        before = arm.r.clone()
        opt = torch.optim.SGD(arm.parameters(), lr=1.0)
        with torch.no_grad():
            arm.alpha.fill_(0.5)
        arm(y, v).pow(2).mean().backward()
        opt.step()
        assert torch.equal(arm.r, before)

    def test_it_is_a_buffer_not_a_parameter(self):
        arm = RandomDir(H, DH, 0)
        assert "r" in dict(arm.named_buffers())
        assert "r" not in dict(arm.named_parameters())


class TestMeanValueDiffersFromRandomOnlyInDirection:
    def test_mean_carries_no_gradient(self):
        y, v = yv(6)
        arm = MeanValue(H, DH)
        arm.train()
        with torch.no_grad():
            arm.alpha.fill_(0.3)
        arm(y, v).pow(2).mean().backward()
        assert arm.m.grad is None
        assert arm.alpha.grad is not None

    def test_ema_moves_only_in_training_mode(self):
        arm = MeanValue(H, DH)
        arm.train()
        arm(*yv(7))
        arm.eval()
        before = arm.m.clone()
        arm(*yv(8))
        arm(*yv(9))
        assert torch.equal(arm.m, before)

    def test_first_batch_initialises_rather_than_decays_toward_zero(self):
        """A cold EMA starting at zero would spend hundreds of steps wrong."""
        arm = MeanValue(H, DH, momentum=0.99)
        arm.train()
        _, v = yv(10)
        arm(torch.zeros_like(v), v)
        expected = v.mean(dim=(0, 2), keepdim=True)
        assert torch.allclose(arm.m, expected, atol=1e-6)


class TestModuleBoundary:
    """The Day-1 gate checks arms.py owns exactly the four output arms."""

    def test_diagmask_is_not_an_output_arm(self):
        assert "diagmask" not in OUTPUT_ARMS
        assert "diagmask" in LOGIT_ARMS

    def test_build_arm_returns_baseline_for_diagmask(self):
        arm = build_arm("diagmask", H, DH, 0)
        assert isinstance(arm, Baseline)

    def test_unknown_arm_raises(self):
        with pytest.raises(KeyError):
            build_arm("nope", H, DH, 0)

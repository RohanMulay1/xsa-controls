"""
xsac.arms - the five interventions.

All arms operate on per-head attention output ``y`` of shape ``(B,H,T,Dh)`` and
per-head values ``v`` of the same shape, **before** ``o_proj``.

Every arm except baseline carries a learnable gate ``alpha`` of shape ``(H,)``
per layer, zero-initialised, applied as ``tanh(alpha)``. Zero-init means every
arm is *exactly* the baseline at step 0, which self-test 10 pins. If that test
fails the entire paired statistical design is void, because the arms would not
start from a common point.

Arm 5 (``diagmask``) is not here. It acts on attention logits rather than on
the attention output, so it lives in the attention module. The Day-1 gate
checks this file defines exactly Baseline, XSA, RandomDir and MeanValue.

Epsilon
-------
XSA's published algorithm divides by ``||v||^2`` with no epsilon. We add one
and say so, here and in the paper. Without it a zero value vector produces a
NaN that would propagate silently through a whole training run.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: XSA's algorithm has no epsilon on ||v||^2. We add one and SAY SO.
EPS = 1e-6


class Arm(nn.Module):
    """Base class. The identity intervention, which is the baseline."""

    name = "base"
    #: Whether this arm exposes a learnable gate. Baseline does not.
    gated = False

    def forward(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return y

    def gate_values(self) -> torch.Tensor:
        """``tanh(alpha)`` per head, or an empty tensor for ungated arms.

        Figure 1 of the paper is this quantity per layer and head. Returning a
        tensor rather than None keeps the recording code uniform.
        """
        return torch.empty(0)


class Baseline(Arm):
    name = "baseline"


class _GatedRankOne(Arm):
    """``z = y - tanh(alpha) * <y, d_hat> d_hat``.

    A rank-one removal along a per-head direction, gated so that the arm can
    learn to switch itself off. Subclasses supply the direction. That single
    choice is the entire difference between the method and its controls, which
    is what makes the comparison clean.
    """

    gated = True

    def __init__(self, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.alpha = nn.Parameter(torch.zeros(n_head))

    def direction(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        d = self.direction(y, v)
        d = d / (d.norm(dim=-1, keepdim=True) + EPS)
        proj = (y * d).sum(-1, keepdim=True) * d
        g = torch.tanh(self.alpha).view(1, -1, 1, 1)
        return y - g * proj

    def gate_values(self) -> torch.Tensor:
        return torch.tanh(self.alpha.detach())


class XSA(_GatedRankOne):
    """Arm 2. The paper's method, gated exactly as modded-nanogpt PR #264.

    Removes the component of the attention output along the token's own value
    vector. Note the position-0 degeneracy: causal softmax over one element
    gives ``a_00 = 1``, so ``y_0 = v_0`` exactly and ``z_0 = 0`` identically in
    every layer and head. Self-test 3 measures it; the paper must state it.
    """

    name = "xsa"

    def direction(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v


class RandomDir(_GatedRankOne):
    """Arm 3. THE CONTROL, and the pre-registered primary endpoint.

    A fixed arbitrary unit direction per (layer, head). Seeded by layer index
    ONLY, deliberately not by the run seed, so the direction is identical
    across all seeds of this arm. We are testing "a fixed arbitrary
    direction", not "a resampled random direction"; those are different
    hypotheses and only the first one isolates rank-one-ness from the choice
    of direction.

    Registered as a buffer so it moves with the model and is never updated by
    the optimiser. Self-test 4 pins both properties.
    """

    name = "random"

    def __init__(self, n_head: int, head_dim: int, layer_idx: int):
        super().__init__(n_head)
        g = torch.Generator().manual_seed(0xA5A5 + 1000 * layer_idx)
        r = torch.randn(n_head, head_dim, generator=g)
        r = r / r.norm(dim=-1, keepdim=True)
        self.register_buffer("r", r.view(1, n_head, 1, head_dim))

    def direction(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.r.expand_as(y)


class MeanValue(_GatedRankOne):
    """Arm 4. The anisotropy control.

    Removes the head's mean value direction, tracked by EMA and detached.
    Detached so that it differs from RandomDir in exactly one respect: which
    direction is removed. If gradients flowed through the mean the two arms
    would differ in optimisation as well as in direction, and the comparison
    would be confounded.

    The EMA updates only in training mode, so evaluation is deterministic and
    two eval forwards with different inputs cannot move the direction.
    Self-tests 5 and 6 pin both halves of that.
    """

    name = "meanval"

    def __init__(self, n_head: int, head_dim: int, momentum: float = 0.99):
        super().__init__(n_head)
        self.momentum = momentum
        self.register_buffer("m", torch.zeros(1, n_head, 1, head_dim))
        self.register_buffer("initialised", torch.zeros(1))

    def direction(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.training:
            with torch.no_grad():
                bm = v.detach().mean(dim=(0, 2), keepdim=True)
                if self.initialised.item() == 0:
                    self.m.copy_(bm)
                    self.initialised.fill_(1)
                else:
                    self.m.mul_(self.momentum).add_(bm, alpha=1 - self.momentum)
        return self.m.expand_as(y)


#: Arms that this module is responsible for. ``diagmask`` is deliberately
#: absent: it modifies attention logits, so the attention module owns it.
OUTPUT_ARMS = ("baseline", "xsa", "random", "meanval")

#: The name the attention module recognises as "mask the diagonal pre-softmax".
LOGIT_ARMS = ("diagmask",)


def build_arm(name: str, n_head: int, head_dim: int, layer_idx: int) -> Arm:
    """Construct an arm by name.

    ``diagmask`` returns a Baseline here, because its intervention happens
    inside attention rather than on the attention output. The caller is
    responsible for passing the arm name into the attention module too. That
    split is checked by ``test_model.py::test_diagmask_is_not_an_output_arm``.
    """
    if name in ("baseline", "diagmask"):
        return Baseline()
    if name == "xsa":
        return XSA(n_head)
    if name == "random":
        return RandomDir(n_head, head_dim, layer_idx)
    if name == "meanval":
        return MeanValue(n_head, head_dim)
    raise KeyError("unknown arm {!r}; expected one of {}".format(
        name, sorted(set(OUTPUT_ARMS) | set(LOGIT_ARMS))))

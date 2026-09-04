"""Apply the XSA rank-one self-value removal to a frozen model, per head.

``xsac.arms.XSA`` defines the intervention for models we train:

    z = y - tanh(alpha) * <y, v_hat> v_hat

This module applies the same operation to a pretrained model we did not
train, one head at a time, so the *measured* effect of the intervention can be
correlated against the statistic that motivates it. The gate is fully open
(``tanh(alpha) = 1``) because we are measuring what the intervention does, not
what a trained gate learns to do with it.

Nothing here is trusted until :func:`verify_reconstruction` passes. It rebuilds
the attention output as ``A @ expand_kv(V)`` from the captured attention matrix
and value vectors and compares it against the tensor the model actually feeds
to its output projection. If those disagree, then the head layout, the KV
expansion or the projection point is wrong, and every downstream number would
be measuring the wrong object while still looking plausible.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .frozen import expand_kv

EPS = 1e-8

#: The output projection, by architecture. Llama/Qwen use ``o_proj``, GPT-2
#: uses ``c_proj`` (a Conv1D), GPT-NeoX uses ``dense``.
OUT_PROJ_NAMES = ("o_proj", "c_proj", "dense", "out_proj")


def out_projection(attn: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    """The module the per-head attention output is fed into."""
    for name in OUT_PROJ_NAMES:
        mod = getattr(attn, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod, name
    raise AttributeError(
        "no output projection on {}; looked for {}".format(
            type(attn).__name__, ", ".join(OUT_PROJ_NAMES)))


def remove_self_value(y: torch.Tensor, v: torch.Tensor,
                      strength: float = 1.0) -> torch.Tensor:
    """``y - strength * <y, v_hat> v_hat``, over the last dimension.

    Identical in form to ``xsac.arms._GatedRankOne.forward`` with the gate
    open. Kept as a free function so the frozen-model path and the trained
    path can be tested against each other rather than reimplemented apart.
    """
    d = v / (v.norm(dim=-1, keepdim=True) + EPS)
    proj = (y * d).sum(-1, keepdim=True) * d
    return y - strength * proj


class HeadIntervention:
    """Hooks one layer so selected heads have their self-value removed.

    Two hooks are needed and they must fire in this order: the attention
    module's pre-hook captures the hidden states so the value projection can
    be recomputed, and the output projection's pre-hook rewrites its input.
    Capturing at the attention module rather than recomputing from the layer
    input matters for models that apply a norm inside the block.
    """

    def __init__(self, probe, layer_idx: int, heads: Optional[Sequence[int]],
                 strength: float = 1.0):
        self.probe = probe
        self.layer_idx = layer_idx
        self.heads = None if heads is None else list(heads)
        self.strength = strength
        self._hidden: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        #: The unmodified input to the output projection, kept for the
        #: reconstruction gate.
        self.captured_y: Optional[torch.Tensor] = None
        #: How many forwards actually had their output rewritten, and how
        #: much the rewrite moved the tensor. A hook that silently fails to
        #: fire produces delta == 0, which is indistinguishable from a real
        #: null result unless it is counted. Callers assert on these.
        self.n_applied = 0
        self.total_change = 0.0

    def _attn(self):
        return self.probe._attn_module(self.probe._layers()[self.layer_idx])

    def _capture_hidden(self, module, args, kwargs):
        if args:
            self._hidden = args[0].detach()
        elif "hidden_states" in kwargs:
            self._hidden = kwargs["hidden_states"].detach()
        return None

    def _rewrite(self, module, args):
        y = args[0]
        self.captured_y = y.detach()
        if self._hidden is None:
            raise InterventionNotApplied(
                "attention input was not captured at layer {}; the "
                "intervention would silently do nothing and report delta 0"
                .format(self.layer_idx))
        nq, nkv = self.probe.head_counts()
        b, t, _ = y.shape
        head_dim = y.shape[-1] // nq
        v = self.probe._values_from_hidden(self.layer_idx, self._hidden)
        if v is None:
            raise InterventionNotApplied(
                "could not recover value vectors at layer {}; the "
                "intervention would silently do nothing and report delta 0"
                .format(self.layer_idx))
        v = expand_kv(v, nq).transpose(1, 2)          # (B, T, nq, D)
        yh = y.view(b, t, nq, head_dim)
        heads = range(nq) if self.heads is None else self.heads
        out = yh.clone()
        for h in heads:
            out[:, :, h, :] = remove_self_value(
                yh[:, :, h, :], v[:, :, h, :].to(yh.dtype), self.strength)
        self.n_applied += 1
        self.total_change += (out - yh).abs().sum().item()
        return (out.view(b, t, nq * head_dim),) + tuple(args[1:])

    def __enter__(self):
        attn = self._attn()
        proj, _ = out_projection(attn)
        self._handles.append(attn.register_forward_pre_hook(
            self._capture_hidden, with_kwargs=True))
        self._handles.append(proj.register_forward_pre_hook(self._rewrite))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._hidden = None
        return False


@contextmanager
def xsa_intervention(probe, layer_idx: int,
                     heads: Optional[Sequence[int]] = None,
                     strength: float = 1.0):
    """Context manager applying XSA to ``heads`` of one layer."""
    with HeadIntervention(probe, layer_idx, heads, strength) as h:
        yield h


def verify_reconstruction(probe, ids: torch.Tensor, layer_idx: int,
                          tol: float = 1e-4) -> Dict[str, float]:
    """Rebuild ``y = A @ expand_kv(V)`` and compare against the real tensor.

    This is a gate, not a diagnostic. If the reconstruction does not match
    what the model feeds to its output projection, the head layout or the KV
    expansion is wrong, every per-head number downstream is measuring
    something other than what it claims, and the failure is silent because the
    numbers still look like numbers.

    Returns the error metrics. Raises :class:`ReconstructionError` past
    ``tol``.
    """
    attn_mod = probe._attn_module(probe._layers()[layer_idx])
    proj, _ = out_projection(attn_mod)

    captured: Dict[str, torch.Tensor] = {}

    def grab_hidden(module, args, kwargs):
        if args:
            captured["hidden"] = args[0].detach()
        elif "hidden_states" in kwargs:
            captured["hidden"] = kwargs["hidden_states"].detach()
        return None

    def grab_y(module, args):
        captured["y"] = args[0].detach()
        return None

    handles = [attn_mod.register_forward_pre_hook(grab_hidden,
                                                  with_kwargs=True),
               proj.register_forward_pre_hook(grab_y)]
    try:
        with torch.no_grad():
            out = probe.model(ids, output_attentions=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    attentions = getattr(out, "attentions", None)
    if not attentions:
        raise ReconstructionError(
            "model returned no attentions; eager attention is required")
    if "y" not in captured or "hidden" not in captured:
        raise ReconstructionError("hooks captured nothing")

    a = attentions[layer_idx].detach().float()        # (B, nq, T, T)
    v = probe._values_from_hidden(layer_idx, captured["hidden"])
    if v is None:
        raise ReconstructionError("could not recover value vectors")
    nq, _ = probe.head_counts()
    v = expand_kv(v, nq).float()                      # (B, nq, T, D)

    rebuilt = torch.matmul(a, v)                      # (B, nq, T, D)
    b, _, t, d = rebuilt.shape
    rebuilt = rebuilt.transpose(1, 2).reshape(b, t, nq * d)
    real = captured["y"].float()

    denom = real.abs().max().item() or 1.0
    max_abs = (rebuilt - real).abs().max().item()
    metrics = {"max_abs_error": max_abs,
               "max_rel_error": max_abs / denom,
               "mean_abs_error": (rebuilt - real).abs().mean().item(),
               "reference_scale": denom,
               "layer": float(layer_idx)}
    if metrics["max_rel_error"] > tol:
        raise ReconstructionError(
            "A @ expand_kv(V) does not reproduce the attention output at "
            "layer {}: max relative error {:.3e} exceeds {:.0e}. The head "
            "layout, the KV expansion or the projection point is wrong; no "
            "per-head number from this model can be trusted.".format(
                layer_idx, metrics["max_rel_error"], tol))
    return metrics


class ReconstructionError(RuntimeError):
    """The captured attention output could not be rebuilt from A and V."""


class InterventionNotApplied(RuntimeError):
    """A hook could not do its work, so the measured delta would be a lie."""


def head_loss_deltas(probe, batches: Sequence[torch.Tensor], layer_idx: int,
                     heads: Iterable[int], strength: float = 1.0
                     ) -> Dict[int, float]:
    """Change in mean next-token loss from removing each head's self-value.

    One head at a time, so the effect is attributable. Positive means the
    intervention made the model worse.
    """
    base = mean_loss(probe, batches)
    out: Dict[int, float] = {}
    for h in heads:
        with xsa_intervention(probe, layer_idx, [h], strength) as hook:
            out[h] = mean_loss(probe, batches) - base
        if hook.n_applied == 0:
            raise InterventionNotApplied(
                "layer {} head {}: the hook never fired, so this delta would "
                "be 0 for the wrong reason".format(layer_idx, h))
        if hook.total_change == 0.0:
            raise InterventionNotApplied(
                "layer {} head {}: the hook fired but changed nothing. Either "
                "the value vectors are zero or the head is already orthogonal "
                "to them; either way a delta of 0 here is not evidence of no "
                "effect".format(layer_idx, h))
    return out


def mean_loss(probe, batches: Sequence[torch.Tensor]) -> float:
    """Mean next-token cross entropy over the batches, in nats."""
    total, count = 0.0, 0
    with torch.no_grad():
        for ids in batches:
            out = probe.model(ids, use_cache=False)
            logits = out.logits[:, :-1].float()
            target = ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
                reduction="sum")
            total += loss.item()
            count += target.numel()
    return total / max(count, 1)

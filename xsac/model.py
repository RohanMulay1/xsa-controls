"""
xsac.model - a GPT with a pluggable output-surgery hook.

Design constraints that come straight from the experiment:

* Arms 2-4 need per-head attention output ``y`` and per-head values ``v``,
  after attention and before ``o_proj``. SDPA returns ``y`` and we already hold
  ``v``, so the fast path stays available for four of the five arms.
* Arm 5 (``diagmask``) needs the explicit attention matrix, so it cannot use
  SDPA. It is ~1.5-2x slower and the budget has to account for that.
* Position 0 under ``diagmask`` is excluded from masking. Its row has no other
  key, so masking the diagonal would give an all -inf row and NaN. This is a
  documented deviation and it must appear in the paper.

Position encoding is RoPE, derived from the spec's own parameter counts; see
``xsac.config``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from xsac.arms import Arm, build_arm
from xsac.config import ModelConfig

#: Logit penalty that stands in for -inf on the gated diagmask path. At full
#: gate strength exp(-30) is 9.4e-14, which is -inf for every practical
#: purpose, while remaining differentiable in the gate.
DIAG_MASK_STRENGTH = 30.0


def build_rope_cache(seq_len: int, head_dim: int, theta: float,
                     device: torch.device,
                     dtype: torch.dtype = torch.float32
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (1, 1, seq_len, head_dim)."""
    if head_dim % 2:
        raise ValueError("RoPE needs an even head_dim, got {}".format(head_dim))
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device,
                                        dtype=torch.float32) / head_dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return (emb.cos()[None, None].to(dtype), emb.sin()[None, None].to(dtype))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> torch.Tensor:
    t = x.shape[-2]
    return x * cos[..., :t, :] + _rotate_half(x) * sin[..., :t, :]


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with an arm applied to the head outputs."""

    def __init__(self, cfg: ModelConfig, arm_name: str, layer_idx: int,
                 diagmask_hard: bool = False):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.arm_name = arm_name
        #: diagmask acts on logits; every other arm acts on the head output.
        self.diagmask = arm_name == "diagmask"
        #: Two diagmask variants exist because the spec describes two.
        #:
        #: Section 5 says "every arm except baseline carries a learnable gate
        #: alpha, zero-initialised", that "zero-init means every arm is
        #: exactly the baseline at step 0", and the Day-1 gate requires
        #: max |loss_arm - loss_baseline| < 1e-6 "across all 5 arms". A hard
        #: -inf diagonal cannot satisfy any of those: measured deviation at
        #: step 0 is 3.4e-3, three orders of magnitude over the gate.
        #:
        #: The gated form subtracts tanh(alpha) * DIAG_MASK_STRENGTH from the
        #: diagonal logit. At alpha = 0 it subtracts exactly zero, so the arm
        #: is the baseline; at full strength it is -inf for all practical
        #: purposes. It is the reading under which the spec is self-consistent
        #: and it is the default.
        #:
        #: ``diagmask_hard`` restores section 5's literal snippet. It is kept
        #: because that snippet is what the paper will describe, and
        #: test_model.py measures and records the step-0 gap it produces
        #: rather than hiding it.
        self.diagmask_hard = bool(diagmask_hard)
        if self.diagmask and not self.diagmask_hard:
            self.diag_alpha = nn.Parameter(torch.zeros(cfg.n_head))
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.arm: Arm = build_arm(arm_name, cfg.n_head, cfg.head_dim, layer_idx)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                capture: Optional[Dict[str, torch.Tensor]] = None
                ) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        shape = (B, T, self.n_head, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.diagmask or capture is not None:
            y, att = self._explicit_attention(q, k, v, mask_diagonal=self.diagmask)
            if capture is not None:
                capture["att"] = att.detach()
        else:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.cfg.dropout if self.training else 0.0)

        y = self.arm(y, v)
        if capture is not None:
            capture["y"] = y.detach()
            capture["v"] = v.detach()

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    def diag_gate_values(self) -> torch.Tensor:
        """``tanh(alpha)`` for the gated diagmask path, else empty."""
        if self.diagmask and not self.diagmask_hard:
            return torch.tanh(self.diag_alpha.detach())
        return torch.empty(0)

    def _explicit_attention(self, q: torch.Tensor, k: torch.Tensor,
                            v: torch.Tensor, mask_diagonal: bool
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The non-SDPA path. Needed by diagmask and by attention capture."""
        T = q.shape[-2]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        causal = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
        att = att.masked_fill(causal, float("-inf"))
        if mask_diagonal and T > 1:
            # Position 0 is excluded. Its row has no other key, so masking the
            # diagonal there gives an all -inf row and NaN out of softmax.
            # This exclusion is a documented deviation from "mask the
            # diagonal" and it belongs in the paper.
            idx = torch.arange(1, T, device=q.device)
            if self.diagmask_hard:
                att[..., idx, idx] = float("-inf")
            else:
                gate = torch.tanh(self.diag_alpha).view(1, -1, 1)
                att[..., idx, idx] = (att[..., idx, idx]
                                      - gate * DIAG_MASK_STRENGTH)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        return att @ v, att


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, arm_name: str, layer_idx: int,
                 diagmask_hard: bool = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg, arm_name, layer_idx,
                                        diagmask_hard=diagmask_hard)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, capture=None):
        x = x + self.attn(self.ln_1(x), cos, sin, capture=capture)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """A GPT whose every layer carries one arm.

    The arm is fixed at construction. Mixing arms across layers would make the
    paired design meaningless, so it is deliberately not supported. PR #264
    applies XSA to only 6 of 12 layers without explaining why; we apply it
    uniformly and say so.
    """

    def __init__(self, cfg: ModelConfig, arm: str = "baseline",
                 diagmask_hard: bool = False):
        super().__init__()
        self.cfg = cfg
        self.arm_name = arm
        self.diagmask_hard = bool(diagmask_hard)
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg, arm, i, diagmask_hard=diagmask_hard)
                                for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight

        cos, sin = build_rope_cache(cfg.block_size, cfg.head_dim,
                                    cfg.rope_theta, torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Scaled init on residual projections, as in GPT-2.
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            n -= sum(p.numel() for m in self.modules()
                     if isinstance(m, nn.LayerNorm) for p in m.parameters())
            n -= sum(a.numel() for a in self.alpha_parameters())
        return n

    def alpha_parameters(self) -> List[torch.Tensor]:
        out = [b.attn.arm.alpha for b in self.h
               if hasattr(b.attn.arm, "alpha")]
        out += [b.attn.diag_alpha for b in self.h
                if hasattr(b.attn, "diag_alpha")]
        return out

    def gate_table(self) -> Dict[str, List[float]]:
        """``tanh(alpha)`` per layer and head. This is Figure 1."""
        out: Dict[str, List[float]] = {}
        for i, b in enumerate(self.h):
            vals = b.attn.arm.gate_values()
            if vals.numel() == 0:
                vals = b.attn.diag_gate_values()
            out["layer_{}".format(i)] = vals.tolist()
        return out

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                captures: Optional[List[Dict[str, torch.Tensor]]] = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError("sequence length {} exceeds block_size {}".format(
                T, self.cfg.block_size))
        cos = self.rope_cos.to(idx.device)
        sin = self.rope_sin.to(idx.device)
        x = self.drop(self.wte(idx))
        for i, block in enumerate(self.h):
            cap = captures[i] if captures is not None else None
            x = block(x, cos, sin, capture=cap)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizer(self, train_cfg) -> torch.optim.Optimizer:
        """AdamW with no weight decay on gains, biases or the arm gates.

        The gates are the quantity Figure 1 reports. Decaying them toward zero
        would bias that figure toward "the arm switched itself off", which is
        one of the outcomes we are trying to measure.
        """
        decay, no_decay = [], []
        alpha_ids = {id(a) for a in self.alpha_parameters()}
        for _, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in alpha_ids or p.dim() < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        groups = [{"params": decay, "weight_decay": train_cfg.weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=train_cfg.lr,
                                 betas=tuple(train_cfg.betas))

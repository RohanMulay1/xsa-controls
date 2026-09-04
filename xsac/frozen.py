"""
xsac.frozen - Track A, frozen-model statistics.

This is the cheap half of the project and the half that reaches XSA's own
scale. Training stops at 124M; frozen measurement runs to 6.9B for about $2.

What it measures, per (model, layer, head):

    cos_self  = mean cos(y_i, v_i)      the statistic XSA's method is built on
    cos_null  = mean cos(y_i, v_j)      j != i, sampled WITHIN the sequence
    excess    = cos_self - cos_null     the part that is actually self-specific

Sampling the null within the sequence matters. Drawing ``v_j`` from another
sequence changes what the null controls for: it would then absorb
between-sequence variation as well as anisotropy, and the excess would be
inflated.

Four implementation constraints, each learned the expensive way:

* **Eager attention, always.** SDPA and flash paths do not expose the
  attention matrix, and a silent fallback would produce a null computed from
  the wrong tensor.
* **One layer at a time.** Retaining ``(L,H,T,T)`` for a 32-layer model at
  T=1024 is about 2 GB in bf16 before activations. Capture per layer, reduce
  to statistics immediately, discard.
* **GQA expansion before anything else.** Under grouped-query attention
  several query heads share one KV head, so "the token's own value vector" is
  shared across a group. The expansion must match HF's ``repeat_kv`` exactly
  or the self/null distinction is silently wrong for every GQA model.
* **Never convert an OOM into a number.** A model that does not fit is
  recorded with status ``oom`` and contributes nothing to any table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


def expand_kv(v: torch.Tensor, n_query_heads: int) -> torch.Tensor:
    """Expand KV heads to query heads, matching HF ``repeat_kv``.

    ``v`` has shape ``(B, n_kv_heads, T, D)``. Under MHA this is the identity.
    Under GQA each KV head is repeated ``n_query_heads // n_kv_heads`` times,
    contiguously, so query head ``h`` uses KV head ``h // n_rep``. The
    contiguity is what makes "within-group" and "across-group" well defined,
    and ``test_frozen.py`` asserts this against HF's own implementation.
    """
    b, n_kv, t, d = v.shape
    if n_query_heads == n_kv:
        return v
    if n_query_heads % n_kv:
        raise ValueError(
            "n_query_heads {} is not a multiple of n_kv_heads {}".format(
                n_query_heads, n_kv))
    n_rep = n_query_heads // n_kv
    return v[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(
        b, n_kv * n_rep, t, d)


def group_of_head(head: int, n_query_heads: int, n_kv_heads: int) -> int:
    """Which KV group a query head belongs to."""
    if n_kv_heads <= 0 or n_query_heads % n_kv_heads:
        raise ValueError("bad head counts: {} query, {} kv".format(
            n_query_heads, n_kv_heads))
    return head // (n_query_heads // n_kv_heads)


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8
            ) -> torch.Tensor:
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


def self_and_null_cosines(y: torch.Tensor, v: torch.Tensor,
                          generator: Optional[torch.Generator] = None,
                          min_position: int = 1
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head ``cos(y_i, v_i)`` and the within-sequence null ``cos(y_i, v_j)``.

    ``y`` and ``v`` are ``(B, H, T, D)`` with ``v`` already GQA-expanded.

    Positions below ``min_position`` are excluded. Position 0 is degenerate:
    causal softmax over one key gives ``a_00 = 1``, so ``y_0 = v_0`` exactly
    and ``cos(y_0, v_0) = 1`` by construction. Including it inflates
    ``cos_self`` with a quantity that carries no information. XSA's own
    Figure 1 restricts its diagonal panel to ``i > 1`` for this reason, which
    is evidence the author knew about it in the diagnostic.

    The null partner ``j`` is drawn uniformly from ``[0, i)`` so it is a
    position the query could actually have attended to. Drawing from the whole
    sequence would include future positions the causal model never sees.
    """
    b, h, t, d = y.shape
    if t <= min_position + 1:
        empty = torch.empty(0, device=y.device)
        return empty, empty
    idx = torch.arange(min_position, t, device=y.device)
    ys = y[:, :, idx, :]
    vs = v[:, :, idx, :]
    cos_self = _cosine(ys, vs)

    # j uniform in [0, i) for each i, so the partner is a legal causal key.
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(0)
    u = torch.rand(b, h, idx.numel(), generator=generator).to(y.device)
    j = (u * idx.float().view(1, 1, -1)).long().clamp(min=0)
    vj = torch.gather(v, 2, j.unsqueeze(-1).expand(b, h, idx.numel(), d))
    cos_null = _cosine(ys, vj)
    return cos_self, cos_null


@dataclass
class HeadStat:
    model: str
    layer: int
    head: int
    cos_self: float
    cos_null: float
    excess: float
    n: int
    kv_group: int = -1
    n_kv_heads: int = -1
    #: A3: cos(y_i, v_i) using a DIFFERENT KV group's value at the same
    #: position. NaN under MHA, where there is no other group to borrow from.
    cos_across_group: float = float("nan")

    def row(self) -> Dict[str, Any]:
        return {"model": self.model, "layer": self.layer, "head": self.head,
                "cos_self": self.cos_self, "cos_null": self.cos_null,
                "excess": self.excess, "n": self.n,
                "kv_group": self.kv_group, "n_kv_heads": self.n_kv_heads,
                "cos_across_group": self.cos_across_group}


class FrozenProbe:
    """Attach to a HuggingFace causal LM and measure Check 1 per head.

    Usage::

        probe = FrozenProbe.from_pretrained("gpt2")
        rows = probe.measure(list_of_token_id_tensors)

    Heavy dependencies are imported lazily so the CPU test suite can import
    this module without transformers installed.
    """

    def __init__(self, model, tokenizer, name: str, device: str = "cpu",
                 dtype: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = device
        self.dtype = dtype
        self.model.eval()

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cpu",
                        dtype: Optional[str] = None,
                        trust_remote_code: bool = False) -> "FrozenProbe":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = None
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16

        tok = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            # Eager is mandatory: SDPA and flash do not expose the attention
            # matrix, and a silent fallback would compute the null from the
            # wrong tensor.
            attn_implementation="eager",
        )
        model.to(device)
        return cls(model, tok, model_id, device=device, dtype=dtype)

    # -- structure discovery -------------------------------------------------

    def _layers(self) -> List[torch.nn.Module]:
        m = self.model
        for path in ("transformer.h", "model.layers", "gpt_neox.layers",
                     "model.decoder.layers", "transformer.blocks"):
            obj = m
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                return list(obj)
        raise RuntimeError(
            "could not locate the decoder layers of {}. Add its path to "
            "FrozenProbe._layers.".format(self.name))

    def head_counts(self) -> Tuple[int, int]:
        """(n_query_heads, n_kv_heads). Equal under MHA."""
        cfg = self.model.config
        nq = (getattr(cfg, "num_attention_heads", None)
              or getattr(cfg, "n_head", None))
        nkv = (getattr(cfg, "num_key_value_heads", None)
               or getattr(cfg, "n_head_kv", None) or nq)
        if nq is None:
            raise RuntimeError("cannot determine head count for {}".format(
                self.name))
        return int(nq), int(nkv)

    def _attn_module(self, layer: torch.nn.Module) -> torch.nn.Module:
        for attr in ("attn", "self_attn", "attention"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise RuntimeError("no attention submodule on {}".format(type(layer)))

    # -- measurement ---------------------------------------------------------

    def _capture_forward(self, ids: torch.Tensor, layers: List[int]):
        """One forward per batch. Returns (attentions, per-layer attn inputs).

        The naive shape of this code runs a separate forward for the attention
        matrix and another for the value projection, per layer. On a 12-layer
        model over 48 documents that is 1152 forwards instead of 48, and it is
        the difference between a minute and an hour on CPU. Hook every layer
        once and reuse the capture.
        """
        hidden: Dict[int, torch.Tensor] = {}
        handles = []
        mods = self._layers()

        def make(idx):
            def cap(module, args, kwargs, output):
                if args:
                    hidden[idx] = args[0].detach()
                elif "hidden_states" in kwargs:
                    hidden[idx] = kwargs["hidden_states"].detach()
            return cap

        for li in layers:
            handles.append(self._attn_module(mods[li]).register_forward_hook(
                make(li), with_kwargs=True))
        try:
            with torch.no_grad():
                out = self.model(ids, output_attentions=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return getattr(out, "attentions", None), hidden

    def _values_from_hidden(self, layer_idx: int, x: torch.Tensor
                            ) -> Optional[torch.Tensor]:
        """Run the layer's value projection on its captured input."""
        attn = self._attn_module(self._layers()[layer_idx])
        b, t, _ = x.shape
        nq, nkv = self.head_counts()
        with torch.no_grad():
            if hasattr(attn, "v_proj"):
                v = attn.v_proj(x)
            elif hasattr(attn, "c_attn"):
                qkv = attn.c_attn(x)
                v = qkv.split(qkv.shape[-1] // 3, dim=2)[2]
            elif hasattr(attn, "query_key_value"):
                qkv = attn.query_key_value(x)
                d = qkv.shape[-1] // (3 * nq)
                v = qkv.view(b, t, nq, 3 * d)[..., 2 * d:].reshape(b, t, nq * d)
            else:
                return None
        head_dim = v.shape[-1] // nkv
        return v.view(b, t, nkv, head_dim).transpose(1, 2)

    def measure(self, batches: Sequence[torch.Tensor],
                layers: Optional[Iterable[int]] = None, seed: int = 0,
                min_position: int = 1,
                layers_per_pass: Optional[int] = None
                ) -> List[Dict[str, Any]]:
        """Measure Check 1 per (layer, head).

        ``layers_per_pass`` bounds memory for large models by re-running the
        forward for each chunk of layers. The spec's memory guard: the full
        (L,H,T,T) tensor for a 32-layer model at T=1024 is ~2 GB in bf16, so
        batch the capture rather than shrinking T. Shrinking T would change
        the statistic being measured, which is not an acceptable trade.
        """
        nq, nkv = self.head_counts()
        n_layers = len(self._layers())
        todo = list(layers) if layers is not None else list(range(n_layers))
        chunks = ([todo] if not layers_per_pass else
                  [todo[i:i + layers_per_pass]
                   for i in range(0, len(todo), layers_per_pass)])

        acc_self: Dict[tuple, List[float]] = {}
        acc_null: Dict[tuple, List[float]] = {}
        acc_across: Dict[tuple, List[float]] = {}

        for chunk in chunks:
            for bi, batch in enumerate(batches):
                ids = batch.to(self.device)
                atts, hidden = self._capture_forward(ids, chunk)
                if not atts:
                    continue
                for li in chunk:
                    if li >= len(atts) or li not in hidden:
                        continue
                    v = self._values_from_hidden(li, hidden[li])
                    if v is None:
                        continue
                    att = atts[li]
                    vx = expand_kv(v, att.shape[1])
                    y = att.to(vx.dtype) @ vx
                    gen = torch.Generator(device="cpu").manual_seed(
                        seed + 1000 * li + bi)
                    cs, cn = self_and_null_cosines(
                        y.float(), vx.float(), generator=gen,
                        min_position=min_position)
                    if cs.numel() == 0:
                        continue
                    # A3: the same position's value taken from a NEIGHBOURING
                    # KV group. Under GQA this asks whether the self-value
                    # similarity is specific to the group's own value or is a
                    # generic property of any value at that position. Under
                    # MHA there is no meaningful "other group", so it stays
                    # undefined rather than being reported as an equal number
                    # that would read as a null result.
                    if 1 < nkv < nq:
                        shifted = torch.roll(v, shifts=1, dims=1)
                        sx = expand_kv(shifted, att.shape[1])
                        ca, _ = self_and_null_cosines(
                            y.float(), sx.float(), generator=gen,
                            min_position=min_position)
                    else:
                        ca = None
                    for h in range(cs.shape[1]):
                        acc_self.setdefault((li, h), []).extend(
                            cs[:, h].flatten().tolist())
                        acc_null.setdefault((li, h), []).extend(
                            cn[:, h].flatten().tolist())
                        if ca is not None:
                            acc_across.setdefault((li, h), []).extend(
                                ca[:, h].flatten().tolist())
                del atts, hidden

        rows: List[Dict[str, Any]] = []
        for (li, h) in sorted(acc_self):
            s = float(np.mean(acc_self[(li, h)]))
            n0 = float(np.mean(acc_null[(li, h)]))
            across = (float(np.mean(acc_across[(li, h)]))
                      if acc_across.get((li, h)) else float("nan"))
            rows.append(HeadStat(
                model=self.name, layer=li, head=h, cos_self=s, cos_null=n0,
                excess=s - n0, n=len(acc_self[(li, h)]),
                kv_group=group_of_head(h, nq, nkv), n_kv_heads=nkv,
                cos_across_group=across).row())
        return rows


def aggregate_model(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Model-level means. Reported with n, never bare."""
    if not rows:
        return {"cos_self": float("nan"), "cos_null": float("nan"),
                "excess": float("nan"), "n_heads": 0,
                "self_specific_fraction": float("nan")}
    cs = float(np.mean([r["cos_self"] for r in rows]))
    cn = float(np.mean([r["cos_null"] for r in rows]))
    return {"cos_self": cs, "cos_null": cn, "excess": cs - cn,
            "n_heads": len(rows),
            "self_specific_fraction": (cs - cn) / cs if cs else float("nan")}


def gqa_within_across(rows, n_query_heads=None):
    """A3: is the self-value statistic specific to the head's own KV group?

    Two things this function must not get wrong, both of which an earlier
    revision did:

    * **GQA is a head-count property, not a group-size one.** It holds when
      ``n_kv_heads < n_query_heads``, full stop. Inferring it from group sizes
      reported GPT-2 (12 query heads, 12 KV heads, plainly MHA) as grouped.
    * **Groups never span layers.** Keying by ``kv_group`` alone put head 3 of
      every layer into one bucket, so a 12-layer MHA model looked like 12
      groups of 12 heads. The key is ``(layer, kv_group)``.

    Under GQA the contrast is ``cos(y_i, v_i)`` using the head's own group's
    value, against the same position's value borrowed from a neighbouring
    group. Under MHA there is no other group, so both are reported undefined
    rather than as two equal numbers that would read as a null result.
    """
    if not rows:
        return {"n_query_heads": 0, "n_kv_heads": -1, "n_groups": 0,
                "is_gqa": False, "n_heads": 0,
                "within_group_excess": float("nan"),
                "across_group_excess": float("nan"), "note": "no rows"}

    n_kv = max(int(r.get("n_kv_heads", -1)) for r in rows)
    heads_per_layer = {}
    for r in rows:
        heads_per_layer.setdefault(int(r.get("layer", 0)), set()).add(
            int(r.get("head", 0)))
    nq = n_query_heads or (max(len(v) for v in heads_per_layer.values())
                           if heads_per_layer else 0)

    is_gqa = bool(0 < n_kv < nq)
    groups = {(int(r.get("layer", 0)), int(r.get("kv_group", -1)))
              for r in rows}
    out = {"n_query_heads": nq, "n_kv_heads": n_kv, "n_groups": len(groups),
           "is_gqa": is_gqa, "n_heads": len(rows),
           "heads_per_kv": (nq // n_kv) if is_gqa and n_kv else 1}
    if not is_gqa:
        out["within_group_excess"] = float("nan")
        out["across_group_excess"] = float("nan")
        out["note"] = ("n_kv_heads == n_query_heads: this model is MHA and "
                       "the within/across split does not exist")
        return out

    def _f(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        return x if math.isfinite(x) else None

    within, across = [], []
    for r in rows:
        cn = _f(r.get("cos_null"))
        cs = _f(r.get("cos_self"))
        cg = _f(r.get("cos_across_group"))
        if cn is None:
            continue
        if cs is not None:
            within.append(cs - cn)
        if cg is not None:
            across.append(cg - cn)
    out["within_group_excess"] = (float(np.mean(within)) if within
                                  else float("nan"))
    out["across_group_excess"] = (float(np.mean(across)) if across
                                  else float("nan"))
    out["note"] = ""
    return out

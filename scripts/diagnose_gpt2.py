"""
Why does GPT-2 not reproduce the reference Check-1 values?

    python scripts/diagnose_gpt2.py

Reference: cos_self 0.5406, cos_null 0.3798, excess 0.1608.
Measured here: 0.4828 / 0.2987 / 0.1840 at block 512.

**This script does not tune toward the reference.** It enumerates the design
choices a Check-1 measurement has to make, states what each one is here,
measures the alternative, and reports every result. If one alternative lands on
the reference that is evidence about which convention the reference used; if
none does, the discrepancy stays unexplained and is reported as unexplained.

The distinction matters because the obvious failure mode is to sweep free
parameters until a target appears and then present the winning setting as the
method. That is the practice this whole project exists to criticise, so the
output below is the full grid, always, with no selection applied.

Choices tested
--------------
1. Sequence length. Already known to matter: the null falls from 0.3747 at
   T=64 to 0.2788 at T=1024 while cos_self barely moves.
2. Position 0. Excluded here, because cos(y_0, v_0) = 1 by construction under
   a causal mask.
3. Null partner. Drawn within the sequence from [0, i) here. The alternatives
   are any j != i in the sequence, or a token from a different sequence.
4. Head aggregation. Per-head means averaged here; the alternative pools all
   heads before averaging.
5. Layer subset. All layers here; the alternative is that the reference used
   some subset.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.runmeta import write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {"cos_self": 0.5406, "cos_null": 0.3798, "excess": 0.1608}
TOL = 0.01


def _cos(a, b, eps=1e-8):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + eps)


@torch.no_grad()
def measure(probe, batches, min_position: int, null_mode: str,
            pool_heads: bool, layers: List[int]) -> Dict[str, float]:
    """One Check-1 measurement under an explicit set of conventions."""
    from xsac.frozen import expand_kv

    self_vals: List[float] = []
    null_vals: List[float] = []
    gen = torch.Generator(device="cpu").manual_seed(0)

    prev_v = None
    for batch in batches:
        ids = batch.to(probe.device)
        atts, hidden = probe._capture_forward(ids, layers)
        if not atts:
            continue
        for li in layers:
            if li >= len(atts) or li not in hidden:
                continue
            v = probe._values_from_hidden(li, hidden[li])
            if v is None:
                continue
            att = atts[li]
            vx = expand_kv(v, att.shape[1]).float()
            y = (att.to(vx.dtype) @ vx).float()
            b, h, t, d = y.shape
            if t <= min_position + 1:
                continue
            idx = torch.arange(min_position, t, device=y.device)
            ys, vs = y[:, :, idx, :], vx[:, :, idx, :]
            cs = _cos(ys, vs)

            if null_mode == "causal":          # j uniform in [0, i)
                u = torch.rand(b, h, idx.numel(), generator=gen).to(y.device)
                j = (u * idx.float().view(1, 1, -1)).long().clamp(min=0)
            elif null_mode == "anywhere":      # any j != i in the sequence
                j = torch.randint(0, t, (b, h, idx.numel()), generator=gen
                                  ).to(y.device)
            elif null_mode == "cross_sequence":
                j = None
            else:
                raise ValueError(null_mode)

            if j is None:
                src = prev_v if prev_v is not None and prev_v.shape == vx.shape \
                    else vx.flip(0)
                vj = src[:, :, idx, :]
            else:
                vj = torch.gather(vx, 2,
                                  j.unsqueeze(-1).expand(b, h, idx.numel(), d))
            cn = _cos(ys, vj)

            if pool_heads:
                self_vals.append(float(cs.mean()))
                null_vals.append(float(cn.mean()))
            else:
                for hh in range(h):
                    self_vals.append(float(cs[:, hh].mean()))
                    null_vals.append(float(cn[:, hh].mean()))
            prev_v = vx
    cs_m = float(np.mean(self_vals)) if self_vals else float("nan")
    cn_m = float(np.mean(null_vals)) if null_vals else float("nan")
    return {"cos_self": cs_m, "cos_null": cn_m, "excess": cs_m - cn_m}


def matches(row: Dict[str, float]) -> bool:
    return all(abs(row[k] - v) <= TOL for k, v in REFERENCE.items())


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-docs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args(argv)

    from xsac.frozen import FrozenProbe
    from scripts.run_frozen import batches_from_docs, load_documents

    docs = load_documents(args.n_docs)
    if not docs:
        print("no real documents available")
        return 1
    probe = FrozenProbe.from_pretrained(args.model, device=args.device)
    n_layers = len(probe._layers())
    all_layers = list(range(n_layers))

    rows: List[Dict[str, object]] = []

    def record(desc, block, **kw):
        batches = batches_from_docs(probe, docs, block, args.n_docs)
        out = measure(probe, batches, layers=kw.pop("layers", all_layers),
                      **kw)
        row = {"variant": desc, "block": block, **out,
               "matches_reference": matches(out)}
        rows.append(row)
        print("  {:44s} T={:4d}  self {:.4f}  null {:.4f}  excess {:.4f}  {}"
              .format(desc, block, out["cos_self"], out["cos_null"],
                      out["excess"], "<-- MATCHES" if row["matches_reference"]
                      else ""))
        return row

    print("reference: self {cos_self}  null {cos_null}  excess {excess}"
          .format(**REFERENCE))
    print("\n1. sequence length (all other conventions as shipped)")
    for t in (64, 128, 256, 512, 1024):
        record("length", t, min_position=1, null_mode="causal",
               pool_heads=False)

    print("\n2. position 0 included rather than excluded")
    record("include position 0", 512, min_position=0, null_mode="causal",
           pool_heads=False)

    print("\n3. null partner definition")
    record("null: any j in sequence", 512, min_position=1,
           null_mode="anywhere", pool_heads=False)
    record("null: token from another sequence", 512, min_position=1,
           null_mode="cross_sequence", pool_heads=False)

    print("\n4. head aggregation")
    record("pool heads before averaging", 512, min_position=1,
           null_mode="causal", pool_heads=True)

    print("\n5. layer subsets")
    half = n_layers // 2
    for name, ls in (("first half", all_layers[:half]),
                     ("second half", all_layers[half:]),
                     ("layer 0 only", [0]),
                     ("last layer only", [n_layers - 1])):
        record("layers: " + name, 512, min_position=1, null_mode="causal",
               pool_heads=False, layers=ls)

    hits = [r for r in rows if r["matches_reference"]]
    print("\n" + "=" * 72)
    if hits:
        print("{} of {} configurations reproduce the reference within "
              "+/-{}:".format(len(hits), len(rows), TOL))
        for h in hits:
            print("   {} at T={}".format(h["variant"], h["block"]))
        print("\nThat identifies which convention the reference used. It does "
              "NOT mean the shipped convention is wrong: the shipped choice is "
              "argued for on its own terms in xsac/frozen.py.")
    else:
        print("NONE of the {} configurations reproduce the reference within "
              "+/-{}.".format(len(rows), TOL))
        print("The discrepancy is not explained by sequence length, position-0 "
              "handling, null-partner definition, head aggregation, or any "
              "layer subset tested. It is reported as unexplained rather than "
              "resolved by further search.")
    print("=" * 72)

    write_csv(rows, ROOT / "results" / "gpt2_diagnosis.csv")
    (ROOT / "results" / "gpt2_diagnosis.json").write_text(
        json.dumps({"reference": REFERENCE, "tolerance": TOL, "rows": rows,
                    "n_reproducing": len(hits),
                    "explained": bool(hits)}, indent=2, default=str),
        encoding="utf-8")
    print("wrote results/gpt2_diagnosis.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

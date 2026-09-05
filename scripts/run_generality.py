"""
A6: Check 1 applied to methods other than XSA.

    python scripts/run_generality.py --model gpt2

This is what makes the paper a checklist rather than a single-method audit.
Each method below motivates an architectural change by pointing at a statistic
and observing that it is large. For each we compute the same thing XSA's audit
computes: the statistic, a null for it that controls for the structure the
statistic inherits for free, and the excess that survives.

Two are implemented here on a frozen causal LM. The spec's guidance is to do
at least two and list the rest as demonstrations of the checklist, which is
what ``NOT_IMPLEMENTED`` records.

Method 1, attention sinks (StreamingLLM, Xiao et al.)
    Statistic: attention mass concentrated on position 0.
    Null:      mass on a matched early position that is not position 0.
    Why:       position 0 is the only position every query can attend to, and
               under a causal mask early positions accumulate mass for
               structural reasons alone. Comparing against position 0 versus
               nothing measures the mask as much as the model.

Method 2, massive activations (Sun et al. 2024)
    Statistic: the largest hidden-state coordinate is enormous.
    Null:      the largest coordinate expected from a matched-variance
               Gaussian of the same width.
    Why:       the maximum of d samples grows like sqrt(2 ln d) even with no
               outlier structure at all. A large max is not by itself evidence
               of a privileged dimension.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.checks import check_null  # noqa: E402
from xsac.runmeta import write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: Listed rather than silently dropped. Each needs a different harness.
NOT_IMPLEMENTED = {
    "value_residual_diff_transformer":
        "needs a matched-capacity control, which means training two models "
        "rather than measuring one frozen model",
    "registers_vit":
        "vision transformers, and the matched control is added tokens; out of "
        "scope for a causal-LM harness",
}


@torch.no_grad()
def attention_sink(probe, batches, min_position: int = 8) -> Dict[str, float]:
    """Mass on position 0 versus a matched early position.

    Only queries at index >= ``min_position`` are counted, so both the
    statistic and its null are measured where several early positions actually
    exist to choose between.
    """
    n_layers = len(probe._layers())
    sink, matched = [], []
    for batch in batches:
        ids = batch.to(probe.device)
        atts, _ = probe._capture_forward(ids, list(range(n_layers)))
        if not atts:
            continue
        for att in atts:
            t = att.shape[-1]
            if t <= min_position + 2:
                continue
            rows = att[..., min_position:, :]
            sink.append(float(rows[..., 0].mean()))
            # The matched null: a fixed early position that is not 0. Averaged
            # over positions 1..3 so the comparison is not to one arbitrary
            # neighbour.
            near = [float(rows[..., j].mean()) for j in (1, 2, 3)]
            matched.append(float(np.mean(near)))
    return {"observed": float(np.mean(sink)) if sink else float("nan"),
            "null": float(np.mean(matched)) if matched else float("nan"),
            "n": len(sink)}


@torch.no_grad()
def massive_activations(probe, batches) -> Dict[str, float]:
    """Largest hidden coordinate versus the matched-variance Gaussian maximum.

    The null is the expected maximum of ``d`` standard normals scaled to the
    observed standard deviation, ``sigma * sqrt(2 ln d)``. Reported in units of
    sigma so the two are directly comparable across models and layers.
    """
    n_layers = len(probe._layers())
    obs, null = [], []
    for batch in batches:
        ids = batch.to(probe.device)
        _, hidden = probe._capture_forward(ids, list(range(n_layers)))
        for _, h in sorted(hidden.items()):
            x = h.float()
            d = x.shape[-1]
            sigma = float(x.std())
            if sigma <= 0:
                continue
            # Per-token maximum absolute coordinate, in units of sigma.
            obs.append(float((x.abs().max(dim=-1).values / sigma).mean()))
            null.append(math.sqrt(2.0 * math.log(d)))
    return {"observed": float(np.mean(obs)) if obs else float("nan"),
            "null": float(np.mean(null)) if null else float("nan"),
            "n": len(obs)}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default=None,
                    choices=[None, "bfloat16", "float16"])
    args = ap.parse_args(argv)

    from xsac.frozen import FrozenProbe
    from scripts.run_frozen import batches_from_docs, load_documents

    docs = load_documents(args.n_docs)
    if not docs:
        print("no real documents available; refusing to report a null "
              "measured on synthetic text")
        return 1

    probe = FrozenProbe.from_pretrained(args.model, device=args.device,
                                        dtype=args.dtype)
    batches = batches_from_docs(probe, docs, args.block, args.n_docs)
    print("model {}  |  {} documents at block {}".format(
        args.model, len(batches), args.block))

    rows: List[Dict[str, object]] = []
    for name, fn, unit, stat_name, null_name in (
            ("attention_sink", attention_sink, "attention mass",
             "mass on position 0", "mass on positions 1-3"),
            ("massive_activations", massive_activations, "sigma",
             "max |h| / sigma", "Gaussian max sqrt(2 ln d)")):
        out = fn(probe, batches)
        # Pass layer indices where the measurement is per-(layer, head), so
        # the interval resamples layers rather than treating heads within a
        # layer as independent draws.
        res = check_null(out["observed"], out["null"], label=name,
                         stat_name=stat_name, null_name=null_name,
                         clusters=out.get("layers"))
        rows.append({
            "method": name, "model": args.model, "unit": unit,
            "observed": out["observed"], "null": out["null"],
            "excess": res.excess,
            "self_specific_fraction": res.self_specific_fraction,
            "survives_null": res.passed, "n": out["n"],
            "status": "completed",
        })
        print("\n" + res.summary())

    for name, why in NOT_IMPLEMENTED.items():
        rows.append({"method": name, "model": args.model, "unit": "",
                     "observed": float("nan"), "null": float("nan"),
                     "excess": float("nan"),
                     "self_specific_fraction": float("nan"),
                     "survives_null": "", "n": 0,
                     "status": "not_run", "reason": why})

    out_path = ROOT / "results" / "generality.csv"
    write_csv(rows, out_path)
    (ROOT / "results" / "generality.json").write_text(
        json.dumps({"model": args.model, "rows": rows,
                    "not_implemented": NOT_IMPLEMENTED}, indent=2, default=str),
        encoding="utf-8")
    print("\nwrote {} ({} implemented, {} listed as future work)".format(
        out_path.name, 2, len(NOT_IMPLEMENTED)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

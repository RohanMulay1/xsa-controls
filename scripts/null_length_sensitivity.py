"""
How much does Check 1's null depend on the sequence length it is measured at?

    python scripts/null_length_sensitivity.py --model gpt2

Motivation. The GPT-2 reference values (cos_self 0.5406, cos_null 0.3798,
excess 0.1608) are quoted without a sequence length. Our measurement did not
reproduce them, so this sweep asks whether length explains the gap.

It partly does, and the answer is a finding in its own right: the null is
strongly length-dependent while the observed statistic is nearly flat, so the
headline "only N% of the statistic is self-specific" is a function of the
measurement context rather than a property of the model. Any paper reporting
Check 1 must state the length it used.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.frozen import FrozenProbe, aggregate_model
from xsac.runmeta import write_csv

ROOT = Path(__file__).resolve().parents[1]
GPT2_TARGET = {"cos_self": 0.5406, "cos_null": 0.3798, "excess": 0.1608}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--lengths", nargs="*", type=int,
                    default=[64, 128, 256, 512, 1024])
    ap.add_argument("--n-docs", type=int, default=24)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    from scripts.run_frozen import load_documents
    docs = load_documents(args.n_docs)
    if not docs:
        print("no real documents available; refusing to report a null "
              "measured on synthetic text")
        return 1

    probe = FrozenProbe.from_pretrained(args.model, device=args.device)
    rows = []
    for T in args.lengths:
        batches = []
        for d in docs:
            ids = probe.tokenizer(d, return_tensors="pt", truncation=True,
                                  max_length=T)["input_ids"]
            if ids.shape[1] >= max(16, T // 2):
                batches.append(ids)
        if not batches:
            continue
        stats = aggregate_model(probe.measure(batches[:args.n_docs]))
        rows.append({"model": args.model, "block_size": T,
                     "n_documents": len(batches[:args.n_docs]), **stats})
        print("  T={:5d}  cos_self {:.4f}  cos_null {:.4f}  excess {:.4f}  "
              "{:.1f}% self-specific".format(
                  T, stats["cos_self"], stats["cos_null"], stats["excess"],
                  100 * stats["self_specific_fraction"]))

    write_csv(rows, ROOT / "results" / "null_length_sensitivity.csv")
    if rows:
        lo = min(r["self_specific_fraction"] for r in rows)
        hi = max(r["self_specific_fraction"] for r in rows)
        print("\nself-specific fraction ranges {:.1f}% to {:.1f}% across "
              "T={} to T={}".format(100 * lo, 100 * hi,
                                    rows[0]["block_size"],
                                    rows[-1]["block_size"]))
        print("The null is context-dependent. Report the length with the "
              "number, always.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

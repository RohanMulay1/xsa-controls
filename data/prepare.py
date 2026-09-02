"""
FineWeb-Edu -> train.bin / val.bin as uint16 memmaps.

    python data/prepare.py --tokens 5.2e8          # TOKENS_PER_RUN x 1.15
    python data/prepare.py --synthetic             # CPU tests, no download

Two guarantees this script provides, both checked afterwards:

* ``train.bin`` is exactly ``2 * n_tokens`` bytes. A truncated write is the
  kind of failure that silently shortens every run's data and shows up as an
  unexplained loss difference weeks later.
* The validation holdout is **disjoint** from training and fixed. Every run,
  arm and seed evaluates on identical batches; shared eval noise would sit
  directly on top of a 0.001 nat effect.

The vocabulary is GPT-2's via tiktoken, with EOT between documents. uint16 is
sufficient because the GPT-2 vocabulary is 50257 < 65536, and the model's
``vocab_size`` of 50304 is that rounded up to a multiple of 64 for kernel
efficiency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent

#: Fixed, disjoint holdout. Identical batches for every run, arm and seed.
VAL_TOKENS = 20_000_000


def prepare_synthetic(out_dir: Path, n_train: int, n_val: int,
                      vocab_size: int) -> dict:
    from xsac.data import synthetic_tokens

    synthetic_tokens(out_dir / "train.bin", n_train, vocab_size, seed=1234)
    synthetic_tokens(out_dir / "val.bin", n_val, vocab_size, seed=99999)
    return {"source": "synthetic", "train_tokens": n_train,
            "val_tokens": n_val, "vocab_size": vocab_size,
            "reportable": False,
            "note": "synthetic tokens for CPU tests only; any run using this "
                    "data must be recorded with status 'smoke'"}


def prepare_fineweb(out_dir: Path, n_train: int, n_val: int,
                    n_proc: int = 8) -> dict:
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)

    def write_split(path: Path, budget: int, stream) -> int:
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(budget,))
        written = 0
        for row in stream:
            ids = enc.encode_ordinary(row["text"])
            ids.append(eot)
            take = min(len(ids), budget - written)
            if take <= 0:
                break
            arr[written:written + take] = np.asarray(ids[:take],
                                                     dtype=np.uint16)
            written += take
            if written >= budget:
                break
        arr.flush()
        del arr
        return written

    it = iter(ds)
    # Validation first, from the head of the stream, then training from what
    # follows. The two never see the same row, so disjointness is structural
    # rather than something an overlap check has to discover.
    n_val_written = write_split(out_dir / "val.bin", n_val, it)
    n_train_written = write_split(out_dir / "train.bin", n_train, it)
    return {"source": "HuggingFaceFW/fineweb-edu sample-10BT",
            "train_tokens": n_train_written, "val_tokens": n_val_written,
            "vocab_size": 50304, "reportable": True,
            "note": "val taken from the head of the stream, train from what "
                    "follows; the splits share no document"}


def verify(out_dir: Path, meta: dict) -> list:
    """Day-1 gate: bytes == 2 * n_tokens, exactly."""
    problems = []
    for split, key in (("train", "train_tokens"), ("val", "val_tokens")):
        p = out_dir / "{}.bin".format(split)
        if not p.exists():
            problems.append("{} is missing".format(p.name))
            continue
        size = p.stat().st_size
        expected = 2 * int(meta[key])
        if size != expected:
            problems.append(
                "{}: {} bytes but {} tokens implies {} bytes".format(
                    p.name, size, meta[key], expected))
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=float, default=5.2e8,
                    help="training tokens; TOKENS_PER_RUN x 1.15")
    ap.add_argument("--val-tokens", type=int, default=VAL_TOKENS)
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--synthetic", action="store_true",
                    help="deterministic synthetic tokens, no download")
    ap.add_argument("--vocab-size", type=int, default=50304)
    ap.add_argument("--n-proc", type=int, default=8)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = int(args.tokens)

    if args.synthetic:
        n_train = min(n_train, 2_000_000)
        n_val = min(args.val_tokens, 400_000)
        meta = prepare_synthetic(out_dir, n_train, n_val, args.vocab_size)
    else:
        meta = prepare_fineweb(out_dir, n_train, args.val_tokens, args.n_proc)

    problems = verify(out_dir, meta)
    meta["verified"] = not problems
    meta["problems"] = problems
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2),
                                       encoding="utf-8")

    print(json.dumps(meta, indent=2))
    if problems:
        print("\nVERIFICATION FAILED:")
        for p in problems:
            print("  - {}".format(p))
        return 1
    print("\nOK: train.bin and val.bin are consistent with their token counts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

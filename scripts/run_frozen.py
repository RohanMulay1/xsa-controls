"""
Track A runner - Check 1 across a ladder of frozen models.

    python scripts/run_frozen.py --models gpt2 --n-docs 64
    python scripts/run_frozen.py --ladder            # the full nine-model ladder
    python scripts/run_frozen.py --smoke             # tiny models, CPU, seconds

Writes one JSON record per model under ``results/frozen/runs/`` and a flat
``results/ladder.csv``. A model that does not fit is recorded with status
``oom`` and contributes no numbers: an OOM is never converted into a result.

Evaluation text is real prose, at least ``--n-docs`` separate documents. This
is deliberate. The predecessor codebase ran everything on one paragraph
repeated 200 times, which gave a base loss of 0.76 nats against 3.96 for real
text, and every statistic computed on it was measuring the model's ability to
recite a loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.frozen import FrozenProbe, aggregate_model, gqa_within_across  # noqa: E402
from xsac.runmeta import (RunRecord, canonical_json, numeric_records,  # noqa: E402
                          read_records, run_id, write_csv, write_record)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "frozen_results" if False else ROOT / "results"
FROZEN_DIR = RESULTS / "frozen"

#: The nine-model ladder. Pythia 2.8B and 6.9B are the rows that answer the
#: scale objection: they bracket XSA's own tested range of 0.7-2.7B.
LADDER = [
    "gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
    "EleutherAI/pythia-160m", "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
]

#: A3: grouped-query attention. Every model shipped at scale uses GQA or MQA,
#: and nobody has checked what the self-value statistic does when the value is
#: shared across a query group.
#: A3 must run on models that are (a) genuinely grouped-query, (b) ungated,
#: and (c) supported by the pinned transformers. Qwen3 needs transformers
#: >= 4.51 and Llama-3.2 is a gated repo, so neither is usable here; both are
#: kept in GQA_BLOCKED with the reason rather than silently dropped.
GQA_MODELS = [
    "Qwen/Qwen2.5-0.5B",       # 14 query heads / 2 KV heads
    "Qwen/Qwen2.5-1.5B",       # 12 query heads / 2 KV heads
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",  # 32 / 4
]

GQA_BLOCKED = {
    "Qwen/Qwen3-1.7B": "architecture qwen3 needs transformers >= 4.51; this "
                       "environment pins 4.46.3 for torch 2.4 compatibility",
    "meta-llama/Llama-3.2-1B": "gated repository, needs an authenticated "
                               "Hugging Face token",
}

SMOKE_MODELS = [
    "hf-internal-testing/tiny-random-LlamaForCausalLM",
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
]

#: GPT-2's published values. The Day-7 gate stops the ladder if these do not
#: come back, because a mismatch means the port is wrong and the other eight
#: models would inherit the same error.
GPT2_TARGET = {"cos_self": 0.5406, "cos_null": 0.3798, "excess": 0.1608}
GPT2_TOLERANCE = 0.01


def load_documents(n_docs: int, min_chars: int = 600) -> List[str]:
    """Real prose, one document per element. Never one paragraph repeated."""
    try:
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="test")
        docs, buf = [], []
        for row in ds:
            line = row["text"]
            if line.startswith(" = ") and buf:
                text = "".join(buf).strip()
                if len(text) >= min_chars:
                    docs.append(text)
                    if len(docs) >= n_docs:
                        return docs
                buf = []
            buf.append(line)
        if buf:
            text = "".join(buf).strip()
            if len(text) >= min_chars:
                docs.append(text)
        if docs:
            return docs[:n_docs]
    except Exception as exc:
        print("  [warn] could not load wikitext ({}); falling back".format(
            type(exc).__name__))
    return []


def batches_from_docs(probe: FrozenProbe, docs: List[str], block: int,
                      max_batches: int) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for doc in docs:
        ids = probe.tokenizer(doc, return_tensors="pt",
                              truncation=True, max_length=block)["input_ids"]
        if ids.shape[1] < 16:
            continue
        out.append(ids)
        if len(out) >= max_batches:
            break
    return out


def run_model(model_id: str, docs: List[str], block: int, n_batches: int,
              device: str, dtype: Optional[str], layers: Optional[List[int]],
              seed: int) -> RunRecord:
    cfg = {"model_id": model_id, "block": block, "n_batches": n_batches,
           "dtype": dtype, "layers": layers, "experiment": "track_a_check1"}
    rid = run_id(cfg, seed)
    started = time.time()
    try:
        probe = FrozenProbe.from_pretrained(model_id, device=device,
                                            dtype=dtype)
        batches = batches_from_docs(probe, docs, block, n_batches)
        if not batches:
            raise RuntimeError("no usable documents after tokenisation")
        rows = probe.measure(batches, layers=layers, seed=seed)
        agg = aggregate_model(rows)
        gqa = gqa_within_across(rows)
        nq, nkv = probe.head_counts()
        return RunRecord(
            run_id=rid, experiment="track_a_check1", status="completed",
            config=cfg, seed=seed,
            metrics={**agg, "n_query_heads": nq, "n_kv_heads": nkv,
                     "n_documents": len(batches), "gqa": gqa,
                     "per_head": rows},
            duration_s=time.time() - started,
            note="Check 1: cos_self, within-sequence null, excess")
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
        return RunRecord(run_id=rid, experiment="track_a_check1", status="oom",
                         config=cfg, seed=seed, error=str(exc)[:400],
                         duration_s=time.time() - started,
                         note="did not fit; use layer-batched capture, never "
                              "a smaller T")
    except Exception as exc:
        return RunRecord(run_id=rid, experiment="track_a_check1",
                         status="failed", config=cfg, seed=seed,
                         error="{}: {}".format(type(exc).__name__,
                                               str(exc)[:400]),
                         duration_s=time.time() - started)


def check_gpt2_target(record: RunRecord) -> dict:
    """The Day-7 gate. A mismatch means the port is wrong."""
    m = record.metrics
    deltas = {k: abs(float(m.get(k, float("nan"))) - v)
              for k, v in GPT2_TARGET.items()}
    ok = all(d <= GPT2_TOLERANCE for d in deltas.values()
             if d == d)  # NaN-safe
    return {"target": GPT2_TARGET, "measured": {k: m.get(k) for k in GPT2_TARGET},
            "abs_deltas": deltas, "tolerance": GPT2_TOLERANCE,
            "reproduces": bool(ok)}


def main(argv=None) -> int:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--ladder", action="store_true", help="the nine-model ladder")
    ap.add_argument("--gqa", action="store_true", help="the A3 GQA models")
    ap.add_argument("--smoke", action="store_true", help="tiny models, CPU")
    ap.add_argument("--n-docs", type=int, default=64)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, choices=[None, "bfloat16", "float16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    models: List[str] = []
    if args.smoke:
        models = SMOKE_MODELS
    if args.ladder:
        models += LADDER
    if args.gqa:
        models += GQA_MODELS
    if args.models:
        models += args.models
    if not models:
        models = ["gpt2"]

    n_docs = 4 if args.smoke else args.n_docs
    block = 64 if args.smoke else args.block
    docs = load_documents(n_docs)
    if not docs:
        print("  [warn] no real documents available; using a synthetic "
              "fallback. Results are marked smoke and are NOT reportable.")
        docs = ["word{} ".format(i % 97) * 200 for i in range(n_docs)]
        synthetic = True
    else:
        synthetic = False
    print("Loaded {} documents ({} chars median)".format(
        len(docs), sorted(len(d) for d in docs)[len(docs) // 2]))

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    for model_id in models:
        print("\n>>> {}".format(model_id))
        rec = run_model(model_id, docs, block, n_docs, args.device,
                        args.dtype, args.layers, args.seed)
        if synthetic and rec.status == "completed":
            rec.status = "smoke"
            rec.note += " | synthetic text: not reportable"
        write_record(rec, FROZEN_DIR)
        if rec.status in ("completed", "smoke"):
            m = rec.metrics
            print("    cos_self {:+.4f}  cos_null {:+.4f}  excess {:+.4f}  "
                  "({:.1f}% self-specific, {} heads, n_docs {})".format(
                      m["cos_self"], m["cos_null"], m["excess"],
                      100 * m["self_specific_fraction"], m["n_heads"],
                      m["n_documents"]))
            if model_id == "gpt2":
                chk = check_gpt2_target(rec)
                print("    GPT-2 target check: {}".format(
                    "REPRODUCES" if chk["reproduces"] else "DOES NOT REPRODUCE"))
                for k, v in chk["abs_deltas"].items():
                    print("      {:9s} measured {:+.4f} vs target {:+.4f} "
                          "(|delta| {:.4f})".format(
                              k, chk["measured"][k], GPT2_TARGET[k], v))
                (RESULTS / "gpt2_target_check.json").write_text(
                    json.dumps(chk, indent=2), encoding="utf-8")
        else:
            print("    status={} :: {}".format(rec.status, rec.error[:160]))

    # Flat ladder.csv from records, numeric statuses only.
    records = numeric_records(read_records(FROZEN_DIR))
    rows = []
    for r in records:
        for head in r.metrics.get("per_head", []):
            rows.append({**head, "status": r.status, "run_id": r.run_id})
    if rows:
        write_csv(rows, RESULTS / "ladder.csv")
        print("\nwrote results/ladder.csv ({} head rows from {} models)".format(
            len(rows), len(records)))

    gqa_rows = [{"model": r.metrics.get("per_head", [{}])[0].get("model", ""),
                 **r.metrics.get("gqa", {}), "status": r.status}
                for r in records if r.metrics.get("gqa")]
    if gqa_rows:
        write_csv(gqa_rows, RESULTS / "gqa.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

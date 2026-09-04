"""A2a and A2: is the per-head XSA effect resolvable, and does the statistic predict it?

A2a runs first and gates A2. It measures each head's intervention effect twice,
on disjoint halves of the evaluation documents, and asks whether the two
estimates agree. If they do not, no correlation computed against that effect
means anything, because the ceiling on any observable correlation is
``sqrt(r_delta * r_stat)``.

A2 then correlates the per-head effect against the statistics that motivate the
method, reporting the raw correlation and the same correlation divided by that
ceiling.

    python scripts/run_reliability.py --models gpt2 EleutherAI/pythia-160m \\
        --n-docs 32 --block 512 --device cuda

Outputs ``results/reliability.csv`` and ``results/a2_correlations.csv``.

Nothing is reported until the A@V reconstruction gate passes: if the attention
output cannot be rebuilt from the captured attention matrix and value vectors,
the head layout is wrong and every per-head number would be measuring the wrong
object while still looking like a number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xsac.checks import check_resolvability, disattenuate  # noqa: E402
from xsac.frozen import FrozenProbe  # noqa: E402
from xsac.intervene import (mean_loss, verify_reconstruction,  # noqa: E402
                            xsa_intervention)
from xsac.runmeta import write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

#: Document budgets for the sample-size sweep. The question is how much
#: evaluation data a per-head effect needs before its ranking is stable.
BUDGETS = (2, 4, 8, 16, 32)
N_REPLICATES = 3


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    from scipy import stats
    if np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(stats.spearmanr(a[ok], b[ok]).correlation)


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def head_deltas(probe, ids, layers):
    """Per-head loss delta from removing that head's self-value.

    One head at a time so the effect is attributable, on one batched forward
    per head so the cost stays linear in heads rather than in heads x docs.
    """
    base = mean_loss(probe, [ids])
    out = {}
    nq, _ = probe.head_counts()
    for li in layers:
        for h in range(nq):
            with xsa_intervention(probe, li, [h]) as hook:
                out[(li, h)] = mean_loss(probe, [ids]) - base
            if hook.n_applied == 0 or hook.total_change == 0.0:
                raise RuntimeError(
                    "layer {} head {}: intervention did not apply; a delta of "
                    "0 here would not be evidence of no effect".format(li, h))
    return out, base


def head_stats(probe, ids, layers, seed=0):
    """cos_self, excess and a_ii per head, on the same documents."""
    rows = probe.measure([ids], layers=layers, seed=seed)
    return {(int(r["layer"]), int(r["head"])): r for r in rows}


def run_model(model_id, n_docs, block, device, dtype, layers_arg, seed=0):
    print("\n=== {} ===".format(model_id))
    probe = FrozenProbe.from_pretrained(model_id, device=device, dtype=dtype)
    nq, nkv = probe.head_counts()
    n_layers = len(probe._layers())
    layers = (list(range(n_layers)) if layers_arg is None
              else [li for li in layers_arg if li < n_layers])
    print("  {} layers x {} query heads ({} KV)".format(n_layers, nq, nkv))

    from run_frozen import batches_from_docs, load_documents
    texts = load_documents(n_docs * 2)
    batches = batches_from_docs(probe, texts, block, n_docs * 2)
    if not batches:
        raise RuntimeError("no usable documents after tokenisation")
    batches = [b if b.dim() == 2 else b.unsqueeze(0) for b in batches]
    # Documents tokenise to slightly different lengths, so they cannot be
    # stacked as they are. Truncate to the shortest rather than padding:
    # padding would put the intervention on positions that carry no signal
    # and dilute every per-head delta by an amount that varies with the
    # document mix.
    common = min(b.shape[1] for b in batches)
    batches = [b[:, :common] for b in batches]
    print("  {} documents truncated to a common {} tokens".format(
        sum(b.shape[0] for b in batches), common))
    all_ids = torch.cat(batches, dim=0).to(probe.device)
    if all_ids.shape[0] < 4:
        raise SystemExit("need at least 4 documents to split into halves")

    # A model whose plain forward is not finite cannot be measured at all.
    # Without this check the NaN deltas propagate into r_delta and the model
    # is reported as UNRESOLVABLE, which says the effect is too noisy to
    # measure. That is a scientific claim, and it would be the wrong one:
    # nothing was measured. Pythia in float32 and float16 on transformers
    # 5.16.1 does exactly this; bfloat16 is fine.
    with torch.no_grad():
        probe_logits = probe.model(all_ids[:1], use_cache=False).logits
    if not torch.isfinite(probe_logits).all():
        raise RuntimeError(
            "{} produces non-finite logits in {} on a plain forward with no "
            "intervention. This is a model/precision failure, not an "
            "unresolvable effect, and it must not be reported as one. Try "
            "--dtype bfloat16.".format(model_id, dtype))

    # The gate. Nothing below is trusted until this passes.
    print("  A@V reconstruction gate:")
    gate = {}
    for li in (0, n_layers // 2, n_layers - 1):
        m = verify_reconstruction(probe, all_ids[:1], li)
        gate["layer_{}".format(li)] = m
        print("    layer {:<3d} relative Frobenius error {:.3e}  PASS".format(
            li, m["rel_frobenius_error"]))

    half = all_ids.shape[0] // 2
    a_ids, b_ids = all_ids[:half], all_ids[half:half * 2]

    print("  measuring per-head effect on two disjoint halves "
          "({} docs each)...".format(half))
    da, base_a = head_deltas(probe, a_ids, layers)
    db, base_b = head_deltas(probe, b_ids, layers)
    sa = head_stats(probe, a_ids, layers, seed)
    sb = head_stats(probe, b_ids, layers, seed)

    keys = sorted(set(da) & set(db) & set(sa) & set(sb))
    delta_a = [da[k] for k in keys]
    delta_b = [db[k] for k in keys]
    stat_a = [float(sa[k]["cos_self"]) for k in keys]
    stat_b = [float(sb[k]["cos_self"]) for k in keys]

    res = check_resolvability(delta_a, delta_b, stat_a, stat_b)
    print("  r_delta = {:+.3f}   r_stat = {:+.3f}   verdict {}".format(
        res.r_delta, res.r_stat, res.verdict.upper()))
    ceiling = float(np.sqrt(max(res.r_delta, 0.0) * max(res.r_stat, 0.0)))
    print("  ceiling on any observable correlation = {:.3f}".format(ceiling))

    # Sample-budget sweep. Rank agreement between two independent halves as a
    # function of how many documents each half contains.
    sweep = []
    rng = np.random.default_rng(seed)
    for budget in BUDGETS:
        if budget * 2 > all_ids.shape[0]:
            continue
        for rep in range(N_REPLICATES):
            idx = rng.permutation(all_ids.shape[0])[:budget * 2]
            x = all_ids[idx[:budget]]
            y = all_ids[idx[budget:]]
            dx, _ = head_deltas(probe, x, layers)
            dy, _ = head_deltas(probe, y, layers)
            ks = sorted(set(dx) & set(dy))
            r = spearman([dx[k] for k in ks], [dy[k] for k in ks])
            sweep.append({"model": model_id, "docs_per_half": budget,
                          "replicate": rep, "rank_agreement": r,
                          "n_heads": len(ks)})
            print("    budget {:>3d} docs  rep {}  rank agreement {:+.3f}"
                  .format(budget, rep, r))

    # A2: correlate the pooled effect against each motivating statistic.
    pooled = [0.5 * (da[k] + db[k]) for k in keys]
    corr_rows = []
    for stat_name in ("cos_self", "excess", "a_ii"):
        if stat_name not in sa[keys[0]]:
            continue
        values = [float(sa[k][stat_name]) for k in keys]
        raw_s = spearman(values, pooled)
        raw_p = pearson(values, pooled)
        corrected = disattenuate(raw_s, res.r_delta, res.r_stat)
        corr_rows.append({
            "model": model_id, "statistic": stat_name,
            "rho_raw": raw_s, "pearson_raw": raw_p,
            "r_delta": res.r_delta, "r_stat": res.r_stat,
            "ceiling": ceiling, "rho_disattenuated": corrected,
            "n": len(keys), "verdict": res.verdict,
            "resolvable": bool(res.passed),
        })
        print("  {:<10s} rho {:+.3f}   ceiling {:.3f}   disattenuated {}"
              .format(stat_name, raw_s, ceiling,
                      "{:+.3f}".format(corrected)
                      if np.isfinite(corrected) else "undefined"))

    rel_row = {
        "model": model_id, "n_heads": len(keys), "docs_per_half": half,
        "block": block, "r_delta": res.r_delta, "r_stat": res.r_stat,
        "ceiling": ceiling, "verdict": res.verdict,
        "resolvable": bool(res.passed), "action": res.action,
        "baseline_loss_half_a": base_a, "baseline_loss_half_b": base_b,
        "gate_max_rel_frobenius": max(m["rel_frobenius_error"]
                                      for m in gate.values()),
        "gate_max_rel_error": max(m["max_rel_error"] for m in gate.values()),
        "status": "completed",
    }
    return rel_row, corr_rows, sweep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+",
                    default=["gpt2", "EleutherAI/pythia-160m",
                             "EleutherAI/pythia-410m"])
    ap.add_argument("--n-docs", type=int, default=32,
                    help="documents per half; twice this many are loaded")
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rel_rows, corr_rows, sweeps = [], [], []
    for model_id in args.models:
        try:
            r, c, s = run_model(model_id, args.n_docs, args.block,
                                args.device, args.dtype, args.layers,
                                args.seed)
            rel_rows.append(r)
            corr_rows.extend(c)
            sweeps.extend(s)
        except Exception as exc:
            print("  FAILED {}: {}: {}".format(
                model_id, type(exc).__name__, str(exc)[:300]))
            rel_rows.append({"model": model_id, "status": "failed",
                             "action": str(exc)[:300]})

    RESULTS.mkdir(exist_ok=True)
    if rel_rows:
        write_csv(rel_rows, RESULTS / "reliability.csv")
        print("\nwrote results/reliability.csv")
    if corr_rows:
        write_csv(corr_rows, RESULTS / "a2_correlations.csv")
        print("wrote results/a2_correlations.csv")
    if sweeps:
        write_csv(sweeps, RESULTS / "reliability_budget_sweep.csv")
        print("wrote results/reliability_budget_sweep.csv")
    (RESULTS / "reliability.json").write_text(
        json.dumps({"reliability": rel_rows, "correlations": corr_rows,
                    "budget_sweep": sweeps}, indent=2, default=str),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

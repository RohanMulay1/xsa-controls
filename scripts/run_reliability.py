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


def run_model(model_id, n_docs, block, device, dtype, layers_arg,
              seed=0, seq_len=None):
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
    # Truncate to a FIXED length, not to the shortest document present.
    # Taking the minimum makes the sequence length a function of how many
    # documents were loaded: at 24 documents per half it came out at 142
    # tokens and at 64 it fell to 137, so raising the evidence also changed
    # the quantity being measured. This repository's own D4 records that
    # Check 1 is strongly length-dependent, so that drift is not harmless.
    # Documents shorter than the target are dropped rather than padded.
    target = seq_len or min(b.shape[1] for b in batches)
    usable = [b[:, :target] for b in batches if b.shape[1] >= target]
    if len(usable) < 4:
        raise RuntimeError(
            "only {} of {} documents reach {} tokens; lower --seq-len or "
            "raise --n-docs".format(len(usable), len(batches), target))
    dropped = len(batches) - len(usable)
    batches = usable
    common = target
    print("  {} documents at a fixed {} tokens ({} too short, dropped)".format(
        sum(b.shape[0] for b in batches), common, dropped))
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

    def stat_halves(name):
        """Both halves of one statistic, or None where it was not measured."""
        if name not in sa[keys[0]] or name not in sb[keys[0]]:
            return None, None
        return ([float(sa[k][name]) for k in keys],
                [float(sb[k][name]) for k in keys])

    # Per-head rows, so the correlation can be looked at rather than only
    # summarised. A single rho hides whether a relationship is driven by a
    # handful of heads, and the scatter is the figure that shows it.
    per_head = []
    for k, dpool in zip(keys, pooled):
        row = {"model": model_id, "layer": k[0], "head": k[1],
               "delta_half_a": da[k], "delta_half_b": db[k],
               "delta_pooled": dpool}
        for stat_name in ("cos_self", "cos_null", "excess", "a_ii"):
            if stat_name in sa[k]:
                row[stat_name + "_half_a"] = float(sa[k][stat_name])
                row[stat_name] = float(sa[k][stat_name])
            if stat_name in sb[k]:
                row[stat_name + "_half_b"] = float(sb[k][stat_name])
        per_head.append(row)
    corr_rows = []
    for stat_name in ("cos_self", "excess", "a_ii"):
        if stat_name not in sa[keys[0]]:
            continue
        va, vb = stat_halves(stat_name)
        if va is None:
            continue
        # Each statistic gets its OWN split-half reliability. Reusing
        # cos_self's for excess and a_ii would disattenuate them by a number
        # measured on a different quantity, and the ceiling reported beside
        # them would not be their ceiling.
        r_stat_own = spearman(va, vb)
        ceiling_own = float(np.sqrt(max(res.r_delta, 0.0)
                                    * max(r_stat_own, 0.0)))
        # Pool both halves of the statistic, matching the pooled delta.
        # Correlating half-A statistic against a two-half delta puts more
        # noise on one side of the pair than the other.
        values = [0.5 * (x + y) for x, y in zip(va, vb)]
        raw_s = spearman(values, pooled)
        raw_p = pearson(values, pooled)
        corrected = disattenuate(raw_s, res.r_delta, r_stat_own)
        corr_rows.append({
            "model": model_id, "statistic": stat_name,
            "rho_raw": raw_s, "pearson_raw": raw_p,
            "r_delta": res.r_delta, "r_stat": r_stat_own,
            "r_stat_cos_self": res.r_stat,
            "ceiling": ceiling_own, "rho_disattenuated": corrected,
            "n": len(keys), "verdict": res.verdict,
            "resolvable": bool(res.passed),
        })
        print("  {:<10s} rho {:+.3f}   r_stat {:+.3f}   ceiling {:.3f}   "
              "disattenuated {}"
              .format(stat_name, raw_s, r_stat_own, ceiling_own,
                      "{:+.3f}".format(corrected)
                      if np.isfinite(corrected) else "undefined"))

    rel_row = {
        "model": model_id, "n_heads": len(keys), "docs_per_half": half,
        "block": block, "seq_len": common, "r_delta": res.r_delta, "r_stat": res.r_stat,
        "ceiling": ceiling, "verdict": res.verdict,
        "resolvable": bool(res.passed), "action": res.action,
        "baseline_loss_half_a": base_a, "baseline_loss_half_b": base_b,
        "gate_max_rel_frobenius": max(m["rel_frobenius_error"]
                                      for m in gate.values()),
        "gate_max_rel_error": max(m["max_rel_error"] for m in gate.values()),
        "status": "completed",
    }
    return rel_row, corr_rows, sweep, per_head


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
    # A smoke run on a tiny model writes the same filenames as a real one.
    # Without somewhere else to put them it silently overwrites committed
    # measurements with two-head toy numbers, which is a data-loss bug, not
    # an inconvenience.
    ap.add_argument("--seq-len", type=int, default=128,
                    help="fixed token length every document is truncated to. "
                         "Fixed on purpose: taking the shortest document "
                         "present makes the sequence length depend on how "
                         "many documents were loaded, and Check 1 is "
                         "length-dependent.")
    ap.add_argument("--results-dir", default=None,
                    help="write outputs here instead of results/. Use it for "
                         "smoke runs so they cannot overwrite real results.")
    args = ap.parse_args(argv)

    global RESULTS
    if args.results_dir:
        RESULTS = Path(args.results_dir)

    rel_rows, corr_rows, sweeps, head_rows = [], [], [], []
    for model_id in args.models:
        try:
            r, c, s, ph = run_model(model_id, args.n_docs, args.block,
                                    args.device, args.dtype, args.layers,
                                    args.seed, args.seq_len)
            rel_rows.append(r)
            corr_rows.extend(c)
            sweeps.extend(s)
            head_rows.extend(ph)
        except Exception as exc:
            print("  FAILED {}: {}: {}".format(
                model_id, type(exc).__name__, str(exc)[:300]))
            rel_rows.append({"model": model_id, "status": "failed",
                             "action": str(exc)[:300]})

    RESULTS.mkdir(exist_ok=True)
    if rel_rows:
        write_csv(rel_rows, RESULTS / "reliability.csv")
        print("\nwrote {}".format(RESULTS / "reliability.csv"))
    if corr_rows:
        write_csv(corr_rows, RESULTS / "a2_correlations.csv")
        print("wrote {}".format(RESULTS / "a2_correlations.csv"))
    if sweeps:
        write_csv(sweeps, RESULTS / "reliability_budget_sweep.csv")
        print("wrote {}".format(RESULTS / "reliability_budget_sweep.csv"))
    if head_rows:
        write_csv(head_rows, RESULTS / "a2_per_head.csv")
        print("wrote {} ({} heads)".format(
            RESULTS / "a2_per_head.csv", len(head_rows)))
    (RESULTS / "reliability.json").write_text(
        json.dumps({"reliability": rel_rows, "correlations": corr_rows,
                    "budget_sweep": sweeps}, indent=2, default=str),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

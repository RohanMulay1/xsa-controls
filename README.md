# xsa-controls

**Sanity checks for attention surgery.** Null models and matched interventions
for representation-motivated architecture design.

A recurring pattern in transformer design: measure a statistic on internal
representations, observe it is large, propose an architectural change that
removes it, report that the change helps. Steps 2 and 4 each need a control and
almost nobody runs either. Step 4 also needs a prior question answered: is the
quantity you are measuring resolvable at all?

This repository ships those three checks as importable code, plus the harness
to run them.

```python
from xsac.checks import check_resolvability, check_null, check_matched
```

| Check | Question | Function |
|---|---|---|
| **0** Resolvability | Is my measurement above its own noise floor, and does it converge as I add samples? | `check_resolvability` |
| **1** Anisotropy null | How much of `cos(y_i, v_i)` survives the cross-term `cos(y_i, v_j)`? | `check_null` |
| **2** Matched intervention | If removing direction `d` helps, does removing an arbitrary direction help too? | `check_matched` |

Run Check 0 first. If the effect is not reliably measurable, every downstream
correlation attenuates toward zero by construction and a null result means
nothing.

---

## Status

**The implementation is complete and validated. The experiments have not been
run.** No GPU has been rented and no budget has been spent.

| | |
|---|---|
| Self-tests | **10/10 pass** on CPU |
| Test suite | **139 tests**, all passing (135 fast + 4 network) |
| Day-1 gate | see `python scripts/verify_day.py 1` |
| Days 2-10 | **not run** — they need the GPU leg |
| Budget spent | **$0.00** of $70 |

Every experiment that has not run is recorded with status `not_run`, `oom` or
`failed` and contributes no numbers to any table or figure. `numeric_records()`
is the only accessor that feeds an aggregate, so an unexecuted cell cannot
become a data point.

### What has actually been measured

All of the following ran on a RunPod RTX 6000 Ada 48GB against real models and
real text (wikitext-103), not synthetic data.

**A1, the scale ladder. Nine models to 6.9B.** This is the experiment that
answers the scale objection, and it is now real data rather than a plan.

| model | params | cos_self | cos_null | excess | % self-specific |
|---|---|---|---|---|---|
| gpt2 | 124M | 0.4828 | 0.2987 | 0.1840 | 38.1% |
| pythia-160m | 160M | 0.4180 | 0.2637 | 0.1544 | 36.9% |
| gpt2-medium | 355M | 0.4252 | 0.2579 | 0.1674 | 39.4% |
| pythia-410m | 410M | 0.4022 | 0.1937 | 0.2086 | 51.9% |
| gpt2-large | 774M | 0.4213 | 0.2117 | 0.2096 | 49.7% |
| pythia-1.4b | 1.4B | 0.3862 | 0.1900 | 0.1963 | 50.8% |
| gpt2-xl | 1.5B | 0.3861 | 0.2069 | 0.1792 | 46.4% |
| pythia-2.8b | 2.8B | 0.3565 | 0.1859 | 0.1705 | 47.8% |
| **pythia-6.9b** | **6.9B** | **0.3404** | **0.1979** | **0.1425** | **41.9%** |

5,408 head-level rows, 32 documents per model.

**The null declines with scale but does not vanish.** `cos_null` falls from
0.2637 at 160M to 0.1979 at 6.9B. Machina & Mercer (NAACL 2024) report that
large Pythia models are isotropic, and that is the sharpest attack on this
paper's framing. The ladder answers it directly: even at 6.9B, **58% of
`cos(y_i, v_i)` is still explained by the null**, and across XSA's own tested
range of 0.7 to 2.7B the self-specific share sits near half (46 to 51%). The
confound does not disappear at the scale the method was actually trained at.

**A3, grouped-query attention.** Nobody had checked what the self-value
statistic does when the value vector is shared across a query group.

| model | query/KV heads | within-group excess | across-group excess |
|---|---|---|---|
| Qwen2.5-0.5B | 14 / 2 | **+0.2415** | **-0.1876** |
| Qwen2.5-1.5B | 12 / 2 | **+0.2731** | **-0.1922** |

The self-value similarity is specific to the head's own KV group. Borrowing a
neighbouring group's value at the same position does not merely lose the
effect, it goes strongly **negative**. GQA models also show a higher
self-specific fraction (56 to 59%) than any MHA model on the ladder (37 to
52%), so the statistic behaves structurally differently under GQA. XSA's Table
1 reports no KV-head count.

**Day-2 calibration, measured on the real machine.**

| quantity | measured |
|---|---|
| throughput, CFG_S | 169,990 tokens/sec |
| achieved | 34.2 TFLOP/s |
| diagmask slowdown | **2.36x** |

The spec predicts a 1.5 to 2.0x slowdown for `diagmask`. The measured 2.36x is
outside that band, so the budget solver weights it accordingly rather than
using the predicted figure.

**A4, XSA's Figure 1 recomputed.** The equicorrelated-value null reproduces the
published floor of 0.200 at an effective attention width of 43 keys and
predicts `cos_self = 0.3682` against the reported 0.373, a gap of 0.005. The
spec's own stated numbers do not close; see `DEVIATIONS.md` D3.

**Check 1 on GPT-2 does not reproduce its reference values**, and the gate is
left failing. Measured 0.4828 / 0.2987 / 0.1840 against 0.5406 / 0.3798 /
0.1608. Sequence length accounts for the null but not the observed statistic;
see `DEVIATIONS.md` D4 and `results/null_length_sensitivity.csv`.

---

## Quick start

```bash
pip install -r requirements.txt

# Day 1, CPU only, no downloads, seconds.
python scripts/selftest_arms.py --json results/selftest.json   # 10/10 PASS
python scripts/a4_recompute.py                                 # pure arithmetic
python scripts/verify_day.py 1

# End to end on synthetic data, CPU, about a minute.
python data/prepare.py --synthetic
python scripts/run_factorial.py --smoke --seeds 42 1337 2024
python scripts/make_figures.py

# Track A against a real frozen model.
python scripts/run_frozen.py --models gpt2 --n-docs 40
python scripts/null_length_sensitivity.py --model gpt2

pytest -m "not slow"      # 135 tests
pytest -m slow            # 4 more, needs network
```

---

## The five arms

All operate on per-head attention output `y` of shape `(B,H,T,Dh)` and per-head
values `v`, **before** `o_proj`. Every arm except baseline carries a learnable
gate `alpha` per head per layer, zero-initialised, so **every arm is exactly
the baseline at step 0**. Measured deviation across all five: `0.000e+00`.

| # | Arm | Direction removed | Tests |
|---|---|---|---|
| 1 | `baseline` | — | control |
| 2 | `xsa` | own `v_i` | reproduce |
| 3 | **`random`** | fixed arbitrary unit vector | **is *any* rank-one removal enough?** |
| 4 | `meanval` | head mean value direction | is it anisotropy? |
| 5 | `diagmask` | `a_ii` masked pre-softmax | is it just self-attention weight? |

**Arm 3 is the experiment.** Everything else is scaffolding. It is the
pre-registered primary endpoint; the other three are secondary with
Holm-Bonferroni correction over that family only.

`random`'s direction is seeded by **layer index only**, never by the run seed,
so it is identical across all seeds of that arm. We test "a fixed arbitrary
direction", not "a resampled random direction" — different hypotheses, and only
the first isolates rank-one-ness from the choice of direction.

### Two things the paper must state

* **We add an epsilon to `||v||^2`. XSA has none.** Without it a zero value
  vector produces a NaN that propagates silently through a whole run.
* **`diagmask` excludes position 0.** Its row has no other key, so masking the
  diagonal there gives an all `-inf` row and NaN. Position 0 is degenerate
  generally: causal softmax over one element gives `a_00 = 1`, so `y_0 = v_0`
  exactly and XSA's `z_0 = 0` identically in every layer and head. Measured:
  `results/position0.txt`. XSA never mentions this, though its own Figure 1
  restricts the diagonal panel to `i > 1`.

---

## The paired protocol

For each seed: identical init across arms, identical data order, identical LR
schedule and eval batches. The only difference between two runs at a seed is
the arm.

The loader is seeded by the seed alone and holds its own generator, so an
unrelated consumer of torch or numpy randomness cannot silently shift the data
order and break the pairing. Validation batches are a fixed deterministic sweep
over a disjoint holdout, identical for every run: shared eval noise would sit
directly on top of a 0.001 nat effect.

Statistic: paired difference vs baseline, one-sample t-test plus Wilcoxon.
**The unpaired std across seeds is always reported alongside**, because the
contrast shows how much noise the pairing removes, and XSA's paper reports
neither.

---

## Repository layout

```
xsac/
  checks.py      THE DELIVERABLE: check_resolvability, check_null, check_matched
  arms.py        the four output arms (diagmask lives in the attention module)
  model.py       GPT with a pluggable output-surgery hook; RoPE
  train.py       one paired run
  calibrate.py   throughput -> token budget against the ACTUAL hourly rate
  frozen.py      Track A: Check 1 per head, GQA expansion, per-layer capture
  stats.py       paired tests, bootstrap, reliability, power, Holm
  figures.py     five figures, each writing its own source CSV
  data.py        uint16 memmap loaders with the pairing guarantee
  runmeta.py     run records, status enum, resumability
scripts/
  selftest_arms.py    the ten self-tests (Day-1 gate)
  a4_recompute.py     XSA Figure 1 under an equicorrelated null
  run_factorial.py    resumable orchestrator, --pilot for GO/NO-GO
  run_frozen.py       the scale ladder
  null_length_sensitivity.py
  make_figures.py     regenerates every figure from results/*.csv
  verify_day.py       machine-checkable day gates
```

---

## Honesty rules this repo enforces in code

Not as prose, as behaviour:

* **An OOM never becomes a number.** Status is an enum and `numeric_records()`
  is the sole accessor for aggregation and plotting.
* **A figure with no data is skipped, never drawn from placeholders.**
  `make_figures.py` reports what it skipped and why.
* **No mean without an `n` and an interval.** `mean_ci` and `paired_test`
  return them in the same dict.
* **Agreement over a degenerate partition is undefined, not 1.0.**
  `replicate_agreement` returns NaN when a threshold collapsed the pool into
  one class, because the reassuring 1.0 it would otherwise print measures
  nothing.
* **A threshold far above the signal is flagged before the experiment runs.**
  `threshold_sanity` prints the ratio; above ~10x the filter is a no-op.
* **An auxiliary term at its analytic bound is flagged.**
  `pinned_at_extremum` — a constant is not a result.
* **Nothing in `results/` is hand-edited.** Every number is produced by a
  script that can be re-run.

These come from a prior campaign where each was learned expensively. See
`DEVIATIONS.md` for what this implementation does differently from the spec,
and why.

---

## Prior art

The null is **standard practice in embedding geometry** (Ethayarajh 2019); we
do not claim to have invented it. The contribution is its absence inside
attention-architecture motivation. Timkey & van Schijndel 2021 is the closest
structural precedent. Machina & Mercer 2024 report large Pythia models are
isotropic, which is the scale attack the A1 ladder answers directly.

**Do not frame anything here as "structure does not predict contribution."**
That is settled and scooped — Serrano & Smith 2019, Jain & Wallace 2019,
Kobayashi 2020, Mohebbi 2023, Hanna et al. 2024, among others. The claim here
is about *matched controls for representation-motivated design*, which none of
them touch.

**Do not write that XSA "omits the anisotropy control."** It is false: XSA's
Figure 1 left plots `cos(v_i, v_j)` at 0.038-0.101. The correct and stronger
statement is that **it measures both ingredients and never combines them**, so
it never reports how much of `cos(y_i, v_i)` is self-specific.

## Scale limitation, stated plainly

We **train** at 51M and 124M. XSA trains at 0.7-2.7B. We **measure** frozen
statistics up to 6.9B, which covers XSA's range, but a frozen statistic is not
a training result. **We cannot and do not claim to refute XSA at its scale.**
The claim is about the mechanism, tested with controls the original omitted.

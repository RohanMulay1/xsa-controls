# Deviations from the specification

Every place the implementation departs from `IMPLEMENTATION.md`, why, and what
was measured. Nothing here was decided to make a test pass. Where the spec
contradicts itself, both readings are implemented and the default is the one
under which the spec is self-consistent.

---

## D1. Position encoding is RoPE. The spec never says.

**Status: derived, not chosen.**

The spec gives `CFG_S` and `CFG_M` as head/layer/width triples plus parameter
counts, but never states the position encoding. The counts settle it:

| | RoPE | learned positions | spec says |
|---|---|---|---|
| CFG_S total | **50.93M** | 51.45M | ~50.9M |
| CFG_S non-embedding | **25.2M** | 25.2M | ~25.2M |
| CFG_M total | **123.59M** | 124.37M | ~124M |
| CFG_M non-embedding | **84.9M** | 84.9M | ~85M |

RoPE hits both totals; learned position embeddings add `block_size * n_embd`
and miss. Learned positions would also make the parameter count change with
context length, which breaks comparability across the ladder.

Pinned by `tests/test_model.py::TestParameterCounts`.

---

## D2. `diagmask` is gated by default. The spec describes it two ways.

**Status: spec contradiction. Both implemented; the self-consistent reading is
the default.**

Three statements in the spec require every non-baseline arm to be exactly the
baseline at step 0:

* §5: "Every arm except baseline carries a learnable gate `α` of shape `(H,)`
  per layer, **zero-initialised**".
* §5: "Zero-init means every arm is *exactly* the baseline at step 0 — a hard
  self-test."
* §16, Day-1 gate: "Test 10 prints max `|loss_arm - loss_baseline|` at step 0
  **across all 5 arms**; < 1e-6".

But §5's illustrative snippet for arm 5 is a hard mask with no gate:

```python
att[..., idx, idx] = float('-inf')
```

**A hard mask cannot satisfy any of the three.** Measured deviation at step 0
with the hard mask, on `CFG_TINY`:

```
baseline   6.217993259
xsa        6.217993259   dev 0.000e+00
random     6.217993259   dev 0.000e+00
meanval    6.217993259   dev 0.000e+00
diagmask   6.214641094   dev 3.352e-03      <- 3350x the 1e-6 gate
```

**Resolution.** The gated path subtracts `tanh(α) * 30` from the diagonal
logit. At `α = 0` it subtracts exactly zero, so the arm is bit-identical to
baseline and test 10 passes at 0.000e+00. At full gate strength `exp(-30)` is
9.4e-14, which is `-inf` for every practical purpose, and the measured
diagonal is 1.0e-13. It is differentiable in `α`, so the arm can learn its own
strength like every other arm.

`GPT(cfg, arm="diagmask", diagmask_hard=True)` restores the literal snippet.
`tests/test_model.py::TestDiagMask` exercises both paths and asserts the hard
path still deviates, so the reason for the default cannot quietly disappear.

Both must be described in the paper: the gated form is what was trained, the
hard form is what the method literally proposes.

---

## D3. The A4 gate's four numbers do not close.

**Status: spec arithmetic error. Reported, not worked around.**

§0.3 and §10/A4 state: floor 0.200, observed 0.373, excess 0.165, 44%. The
Day-1 gate requires reproducing all four. They are mutually inconsistent:

```
0.373 - 0.200 = 0.173,  not 0.165          (differs by 0.008)
0.173 / 0.373 = 46.4%,  not 44%
```

The excess of 0.165 and the 44% agree with each other and with observed 0.373,
but they require a floor of **0.208**, not 0.200.

`scripts/a4_recompute.py` implements the equicorrelated-value null from first
principles and reports every consistent reading rather than picking one
silently. The model is a good one: solving for the effective attention width
that reproduces the published floor of 0.200 gives `n_eff = 43.1`, at which
the model **predicts** `cos_self = 0.3682` against the reported 0.373, a gap
of 0.005.

The qualitative claim is unaffected and is what matters: the self-specific
fraction is 45-46% across a 16x range of `n_eff`, so well under half of
`cos(y_i, v_i)` survives an anisotropy null, on XSA's own numbers.

The Day-1 gate here checks the recompute exists, that its own arithmetic
closes, and that it **detected** the inconsistency. Adjusting a number to make
the stated gate pass would be the anti-pattern the spec's own table forbids.

---

## D4. GPT-2 does not reproduce the reference Check-1 values.

**Status: does not reproduce. Diagnosed, not tuned.**

The Day-7 gate requires GPT-2 to return `0.5406 / 0.3798 / 0.1608` within
±0.01, and says to stop and fix the port if it does not. Measured on 40
wikitext-103 documents at T=384:

| | measured | target | delta |
|---|---|---|---|
| cos_self | 0.4832 | 0.5406 | 0.057 |
| cos_null | 0.3026 | 0.3798 | 0.077 |
| excess | 0.1807 | 0.1608 | 0.020 |

**The port is not obviously wrong.** The GQA expansion matches HF `repeat_kv`
bit for bit, attention is forced eager, the reconstruction `y = att @ v`
agrees with the module's own output, and the probe behaves correctly across
three model families.

**What the gap is:** sequence length, which the reference values are quoted
without. `results/null_length_sensitivity.csv`:

| T | cos_self | cos_null | excess | % self-specific |
|---|---|---|---|---|
| 64 | 0.4900 | **0.3747** | 0.1153 | 23.5% |
| 128 | 0.4846 | 0.3361 | 0.1485 | 30.7% |
| 256 | 0.4812 | 0.3096 | 0.1716 | 35.7% |
| 512 | 0.4772 | 0.2904 | 0.1868 | 39.2% |
| 1024 | 0.4743 | 0.2788 | 0.1955 | 41.2% |

At T=64 the null is 0.3747 against the target 0.3798, inside the ±0.01
tolerance. `cos_self` stays near 0.48 at every length and never reaches
0.5406, so length explains the null but not the observed statistic. The
remaining gap is likely a different layer subset, model variant, or corpus;
none of those is recoverable from the numbers as quoted.

**This is a finding, not only a discrepancy.** The null is strongly
length-dependent while the observed statistic is nearly flat, so the
self-specific fraction moves from **23.5% to 41.2%** across a 16x length
range. The headline "only N% of the statistic is self-specific" is a property
of the measurement context as much as of the model. Any paper reporting
Check 1 must state the sequence length alongside the number, and the same
sensitivity appears analytically in D3's `n_eff` sweep.

The gate is left **failing** rather than retuned. Chasing free parameters until
a target number appears is precisely the practice this project exists to
criticise.

---

## D5. GQA detection is a head-count property.

**Status: implementation bug found and fixed during validation.**

An earlier revision inferred grouping from group sizes and keyed groups by
`kv_group` alone. Both were wrong, and together they reported GPT-2 (12 query
heads, 12 KV heads, plainly MHA) as `is_gqa=True` with 12 groups, because head
3 of all 12 layers landed in one bucket. It also reported `within_group_excess`
and `across_group_excess` as the same number, which reads as a null result
rather than as an undefined one.

Now: GQA is `n_kv_heads < n_query_heads`, groups are keyed `(layer, kv_group)`,
and under MHA both columns come back NaN with a note saying the split does not
exist. The across-group statistic is `cos(y_i, v_i)` with the value borrowed
from a neighbouring KV group, which is a real contrast rather than a relabelled
copy of the within-group one.

---

## D6. Zero-variance paired differences return a finite verdict.

**Status: edge case, handled explicitly.**

If every seed moves by an identical amount the paired sd is exactly 0, the t
statistic diverges, and the naive implementation returns NaN for what is
actually the strongest evidence the design can produce. `paired_test` now
returns the correct limit (`t = ±inf`, `p = 0`) and sets `zero_variance` with a
note warning that, at small n, an exactly constant difference more often means
the arms were not independent than that the effect is certain.

---

## D7. Track A measurement does one forward per batch, not per layer.

**Status: performance, no change to results.**

The natural implementation runs one forward for the attention matrix and
another for the value projection, per layer. On a 12-layer model over 48
documents that is 1152 forwards rather than 48. All layers are hooked once and
the capture is reused. `layers_per_pass` bounds memory for large models by
re-running per chunk, which is the spec's own guidance: batch the capture,
never shrink T, because shrinking T changes the statistic being measured (see
D4).

## D8. The spec's prior A2 correlations came from degenerate input.

**Status: prior values superseded by measurement on real prose.**

Section 10/A2 quotes prior GPT-2 values for the correlation between the
motivating statistic and the measured per-head intervention effect:
**Spearman 0.043 / 0.017 / -0.021** for `cos_self` / `excess` / `a_ii`. Read
as a target, those numbers say the statistic is essentially unrelated to
where the intervention helps, on every statistic, and that a measurement
returning anything larger is suspect.

They are not a usable reference. The spec's own section 17 lists four bugs in
the `overlap-vs-contribution` code they came from, and the second is decisive
here:

> **Degenerate input.** Everything runs on `SAMPLE_TEXT * 200` -- one
> paragraph x200, base loss 0.76 nats vs 3.96 for real prose. Fix: >= 50 real
> documents.

A correlation across heads needs variation across heads to correlate. One
paragraph repeated two hundred times gives a model almost nothing to do
differently in different heads: at 0.76 nats it is close to memorising a
short cycle. Near-zero correlations on that input are what the input
produces, not a property of the method.

**Measured here, on real wikitext-103 documents** (`results/a2_correlations.csv`,
64 documents per half, disjoint halves, 144 to 384 heads per model):

| model | `cos_self` rho | `excess` rho | verdict |
|---|---|---|---|
| gpt2 | **+0.469** | +0.236 | reliable |
| pythia-160m | +0.014 | **+0.487** | attenuated |
| pythia-410m | +0.099 | **+0.216** | attenuated |

On GPT-2 the raw statistic correlates at **+0.469**, an order of magnitude
above the quoted 0.043, and the relationship is real rather than noise: the
split-half reliability of the effect is +0.799, so the ceiling on any
observable correlation is 0.892 and +0.469 sits well inside it. Two further
runs at a smaller evaluation budget put it at +0.450 and +0.416, so the
tenfold gap against the prior figure is not a one-run artifact.

**The ordering also reverses across models,** which no single prior number
could have expressed. On both Pythia models the raw statistic carries
essentially nothing (+0.001, +0.039) while the null-corrected excess carries
most of it; on GPT-2 the reverse. We report the disagreement rather than the
average, and the claim we make is correspondingly narrow: whether a raw
statistic predicts its own intervention is model-dependent.

**Why this deviation is recorded rather than silently corrected.** A reader
comparing this repository against the spec will find a tenfold discrepancy on
the number the whole of A2 turns on. Without this entry the obvious
conclusion is that our measurement is wrong. The prior value is not a
measurement of the quantity it appears to describe, and the fix is the one
the spec itself prescribes: real documents.

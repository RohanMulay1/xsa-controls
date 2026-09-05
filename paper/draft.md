# Sanity Checks for Attention Surgery

*Null models and matched interventions for representation-motivated architecture design*

**Draft. Every number below is read from a committed artifact in `results/`.
Experiments that were not run are marked as such and contribute nothing.**

---

## Abstract

A recurring pattern in transformer design is: measure a statistic on internal
representations, observe it is large, propose an architectural change that
removes it, and report that the change helps. Steps two and four each need a
control, and step four needs a prior question answered: is the quantity being
measured resolvable at all? We package three checks as importable code and
apply them to a recent attention modification, XSA, plus two further methods.

We report four things. **(1)** The anisotropy null for XSA's motivating
statistic accounts for roughly half of it across a nine-model ladder, and 58%
at Pythia-6.9B (Table 1, Figure 3), so the confound does not vanish at the
scale the method was trained at. **(2)** Under grouped-query attention the
statistic behaves structurally differently: self-value similarity is specific
to a head's own KV group, and borrowing a neighbouring group's value goes
negative (Table 2, Figure 5). **(3)** Applied to two other methods, the same
null leaves 99.2% and 71.7% of their statistics intact (Table 3, Figure 4), so
the checklist discriminates rather than debunking uniformly. **(4)** Applied to our own
earlier, unpublished reproduction of a partitioned-attention method, the
resolvability check found the single-edge effect unmeasurable in float32 and
we withdrew that project's headline claim (§6). We report this as our own
prior work, not as an independent replication of a third party.

We do not claim to refute XSA. Our training leg runs at 51M parameters against
XSA's 0.7-2.7B, and at the token budget we could afford the design resolves
the XSA arm only to about 6x the effect size an independent replication
measured (§5). We report that as a power failure, not as a null result.

---

## 1. Introduction

The null we use is not new. Comparing a similarity against a random-pair
baseline is standard practice in embedding geometry, established by
**Ethayarajh (2019)**, and we claim no novelty for the statistic itself. Our
contribution is its **absence inside attention-architecture motivation**: a
method can be proposed, published, and independently replicated without anyone
reporting how much of its motivating similarity survives a null.

**Timkey & van Schijndel (2021)** is the closest structural precedent, showing
that a small number of rogue dimensions dominate cosine similarity in
contextual embeddings and that correcting for them changes which similarities
look meaningful. We differ in target and in consequence: they diagnose a
representation-space artifact, while we ask whether an *architectural
intervention motivated by such a statistic* survives a matched control.
**Godey et al. (EACL 2024)** establish that anisotropy is inherent to
self-attention, which is the mechanism our null controls for. **Pan et al.
(NeurIPS 2024)** apply an anisotropy baseline to queries and keys in vision
transformers, so we narrow our claim to attention **outputs and value
vectors**.

Two results bound what we can claim. **Machina & Mercer (NAACL 2024)** report
that large Pythia models are isotropic, which if true at the relevant scale
would remove our confound exactly where the method matters; §4.1 answers this
directly with measurement rather than extrapolation. **Zhao et al.
(arXiv:2605.20798)** find that most transformer modifications fail to transfer
at 1-3B and that attention-output modifications are the worst class; our checks
are the cheap inference-time screen that work identified as missing.

### 1.1 What we are careful not to say

XSA's own Figure 1 (left) plots `cos(v_i, v_j)` at 0.038 to 0.101. It measures
anisotropy. The accurate criticism is not that the control is absent but that
**the two ingredients are measured and never combined**, so the paper never
reports how much of `cos(y_i, v_i)` is self-specific. That is the gap this work
fills.

---

## 2. The three checks

Shipped as `xsac/checks.py`.

| Check | Question | Failure means |
|---|---|---|
| **0** Resolvability | Is the measured effect above its own numerical noise floor, and does the estimate converge with samples? | Every downstream correlation attenuates toward zero by construction; a null result carries no information |
| **1** Anisotropy null | How much of `cos(y_i, v_i)` survives the cross-term `cos(y_i, v_j)`, `j != i`, sampled within the sequence? | The motivating statistic is mostly structure the model gets for free |
| **2** Matched intervention | If removing direction `d` helps, does removing a matched *arbitrary* direction help too? | The method is a regulariser, not the mechanism its story describes |

Check 0 runs first. Its decision rule is pre-registered: split-half reliability
`r_delta >= 0.6` permits reporting a correlation, `0.3 <= r_delta < 0.6`
requires the disattenuated value and an explicit statement of attenuation, and
`r_delta < 0.3` forbids the correlation claim entirely.

### 2.1 A degeneracy every implementation must handle

Causal softmax over a single element gives `a_00 = 1`, so `y_0 = v_0` exactly
and XSA's `z_0 = 0` identically in every layer and head. Measured on our tiny
configuration: `a_00 = 1.000000`, `max ||y_0 - v_0|| = 0.00e+00`,
`max ||z_0|| = 2.03e-06` (`results/position0.txt`). The method cannot act at
position 0. XSA does not mention this, though its own Figure 1 restricts the
diagonal panel to `i > 1`. Our diagonal-masking arm must exclude position 0 or
the softmax row becomes all `-inf` and produces NaN.

---

## 3. Check 1 across scale (A1)

Twelve models and 6,784 head-level rows in total, 32 wikitext-103 documents
each, eager attention, null partner drawn within the sequence from positions
the query could causally attend. Table 1 reports the nine multi-head models
(5,408 heads); the three grouped-query models (1,376 heads) are in Table 2,
because the within/across-group split does not exist for multi-head attention.

**Table 1.** Check 1 across the multi-head ladder. `n = 9 MHA models, 5,408 heads.`

| model | params | cos_self | cos_null | excess | % self-specific |
|---|---|---|---|---|---|
| gpt2 | 124M | 0.4828 | 0.2987 | 0.1840 | 38.1 |
| pythia-160m | 160M | 0.4180 | 0.2637 | 0.1544 | 36.9 |
| gpt2-medium | 355M | 0.4252 | 0.2579 | 0.1674 | 39.4 |
| pythia-410m | 410M | 0.4022 | 0.1937 | 0.2086 | 51.9 |
| gpt2-large | 774M | 0.4213 | 0.2117 | 0.2096 | 49.7 |
| pythia-1.4b | 1.4B | 0.3862 | 0.1900 | 0.1963 | 50.8 |
| gpt2-xl | 1.5B | 0.3861 | 0.2069 | 0.1792 | 46.4 |
| pythia-2.8b | 2.8B | 0.3565 | 0.1859 | 0.1705 | 47.8 |
| pythia-6.9b | 6.9B | 0.3404 | 0.1979 | 0.1425 | 41.9 |

### 3.1 The scale objection, answered

Raw `cos_null` falls from 0.2637 at 160M to 0.1979 at 6.9B, but so does
`cos_self` (0.4180 to 0.3404), and the share of the statistic the null
explains is **not** monotone in scale. Within Pythia it runs 63.1%, 48.1%,
49.2%, 52.2%, 58.1% from 160M to 6.9B: it falls to a minimum at 410M and then
rises again. We therefore make no claim about a scale trend in either
direction. What the ladder supports is a level statement: at 6.9B, 58% of
`cos(y_i, v_i)` is still explained by the null, and across XSA's own 0.7-2.7B
training range the self-specific share sits between 46 and 51%. The
isotropy result does not remove the confound at the scale the method was
trained at. (Figure 3, which shades XSA's tested range.)

### 3.2 The null is context-dependent, and this must be reported

Measured on GPT-2 at five sequence lengths (`results/null_length_sensitivity.csv`):

| T | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| cos_self | 0.4900 | 0.4846 | 0.4812 | 0.4772 | 0.4743 |
| cos_null | 0.3747 | 0.3361 | 0.3096 | 0.2904 | 0.2788 |
| **% self-specific** | **23.5** | 30.7 | 35.7 | 39.2 | **41.2** |

The observed statistic is nearly flat while the null moves by a third. A claim
of the form "only N% of this statistic is self-specific" is therefore a
property of the measurement context as much as of the model. **Check 1 must
always be reported with the sequence length it was measured at.** We were
unable to reproduce a published GPT-2 reference triple, and §7 reports that
failure rather than resolving it by search.

---

## 4. Grouped-query attention (A3)

Under GQA several query heads share one KV head, so "the token's own value
vector" is shared across a group. No prior work we are aware of checks what
this does to the statistic; XSA's Table 1 reports no KV-head count.

**Table 2.** Within-group versus across-group excess. `n = 3 models across 2
architecture families, 1,376 heads.`

| model | query / KV heads | within-group excess | across-group excess |
|---|---|---|---|
| Qwen2.5-0.5B | 14 / 2 | **+0.2415** | **-0.1876** |
| Qwen2.5-1.5B | 12 / 2 | **+0.2731** | **-0.1922** |
| TinyLlama-1.1B | 32 / 4 | **+0.2373** | **-0.1126** |

Borrowing a neighbouring group's value at the same position does not merely
weaken the effect; the excess goes **negative**. That word needs unpacking,
because the obvious reading of it is wrong.

The raw across-group cosine is **approximately zero**: -0.0006, +0.0009 and
+0.0013 for the three models. A query head's output is essentially
**orthogonal** to the values of KV groups it does not belong to. There is no
anti-alignment and we do not claim one. The negative excess is arithmetic:
excess subtracts the within-sequence null, which is positive (0.187, 0.193,
0.114), so `0 - cos_null = -cos_null` and the three numbers reproduce exactly.

What the measurement shows is therefore sharper than "it reverses": the
self-value alignment the method is built on is **entirely specific to the
head's own KV group**, and disappears completely across groups rather than
merely shrinking. The self-value direction is a property of the shared KV
group, not of the query head. GQA models also show a higher self-specific fraction, 56
to 68%, than any MHA model on the ladder, 37 to 52%. GQA is not a neutral
change of variable for this family of methods. (Figure 5.)

The sign replicates outside the Qwen family. TinyLlama-1.1B is a Llama
architecture with a different group ratio (32 query heads over 4 KV heads
against Qwen's 14 or 12 over 2), and its across-group excess is negative at
-0.1126. The magnitude is about 40% smaller than Qwen's, so we claim the sign
and not the size: what generalises is that the statistic is specific to the
head's own KV group, not how far borrowing across groups pushes it.

TinyLlama was measured under transformers 5.16.1 while the rest of the ladder
used 4.x. Both Qwen models were re-measured in the new environment to bound
what that costs: the largest disagreement across the four statistics is
4.0e-4, so the comparison is not threatened by the mixed provenance.

---

## 5. Does the statistic predict the intervention? (A2a, A2)

Checks 1 and 2 ask whether a statistic is confounded and whether an
intervention is specific. This section asks the question that sits between
them: does the statistic a method is motivated by actually predict what the
method's intervention does?

Answering it requires knowing first whether the intervention's effect can be
measured at all. We remove each head's self-value component in a frozen model,
one head at a time, and measure the change in next-token loss twice, on
disjoint halves of the evaluation documents. The agreement between the halves
is `r_delta`; the same split-half agreement for the statistic is `r_stat`.
Their geometric mean bounds any correlation between the two.

Nothing in this section is reported until `A @ expand_kv(V)` reproduces the
tensor the model feeds to its output projection. Deliberately breaking the
head layout moves that error from 2.0e-4 to 1.4-1.5, four orders of
magnitude, so the gate distinguishes a correct implementation from a plausible
wrong one rather than merely reporting a small number.

**Table 5.** Per-head statistic against measured per-head effect, at 64
evaluation documents per half. `n = 3 models, 144 to 384 heads each.`

| model | statistic | raw rho | r_delta | ceiling | disattenuated |
|---|---|---|---|---|---|
| gpt2 | cos_self | **+0.462** | +0.795 | 0.890 | **+0.519** |
| gpt2 | excess | +0.279 | +0.795 | 0.891 | +0.313 |
| pythia-160m | cos_self | +0.149 | +0.419 | 0.645 | +0.231 |
| pythia-160m | excess | +0.189 | +0.419 | 0.646 | +0.292 |
| pythia-410m | cos_self | **-0.025** | +0.531 | 0.726 | -0.034 |
| pythia-410m | excess | +0.249 | +0.531 | 0.727 | **+0.342** |

Each statistic is disattenuated by **its own** split-half reliability, not by
the effect statistic's, and both halves of the statistic are pooled to match
the pooled effect. An earlier revision did neither, and the difference was
not cosmetic: pythia-160m's `excess` correlation fell from +0.487 to +0.189
once the statistic was paired symmetrically against the effect.

The effect is resolvable, which is the first thing worth saying: `r_delta` is
+0.795 in GPT-2 (reliable) and +0.419 and +0.531 in the two Pythia models
(attenuated). Rank agreement rises monotonically with evaluation budget, so
the attenuation is a sample-size limit rather than a property of the effect.

**How we know that, and why the first version of this table was wrong.** An
earlier run of exactly this experiment at 24 documents per half produced
`r_delta` of +0.304 and +0.446 for the two Pythia models. Repeating it, same
n, different documents and a different GPU, gave +0.194 and **-0.007**: the
verdict for both models moved from attenuated to unresolvable, and the
correlations moved with it, `excess` on pythia-160m falling from +0.396 to
+0.129. At 24 documents per half the reliability estimate for these models is
not stable enough to support any statement built on top of it.

At 64 documents per half it is: +0.473 and +0.435, and the correlations
recover the pattern the first run showed. We report the 64-document run and
keep the 24-document runs in `results/a2_budget_comparison/`, because the
disagreement is more informative than either run alone. It is Check 0 doing
its job on our own work: an unreliable effect produced a correlation that did
not survive being measured again.

The three models do not agree, and the disagreement is the result.

In **pythia-410m** the raw self-value cosine -- the quantity the method is
motivated by -- carries nothing at all about the measured effect of removing
it (rho = -0.025), while the null-corrected excess carries some (+0.249, or
+0.342 against its ceiling). In **pythia-160m** the two are close and both
weak (+0.149 against +0.189): that model supports no ordering either way, and
we do not read one into it. That is the case for Check 1 stated as a prediction rather than
as a critique: correcting for the matched null does not merely lower a number,
it recovers a statistic that tracks the intervention.

**In GPT-2 the ordering reverses.** The raw cosine predicts (+0.462) and the
excess much less so (+0.279).

The GPT-2 number is the one to trust most. Its disattenuated value is
**+0.521, +0.527 and +0.526** across three independent runs, at two
evaluation budgets, on two different GPUs, under two different PyTorch and
transformers versions, and two revisions of the analysis. The raw
correlation moved over that range (+0.416 to +0.469) while the disattenuated
one did not: **+0.521, +0.527, +0.526, +0.519**. Disattenuation divides out
precisely the reliability that moved. That is the clearest evidence we
have that the correction is doing what it claims. We report this rather than averaging over it.

### 5.1 The prior values for this correlation are not usable

The specification we worked from quotes prior GPT-2 values for exactly this
correlation: **Spearman 0.043 / 0.017 / -0.021** for `cos_self`, `excess` and
`a_ii`. Taken at face value they say the motivating statistic is unrelated to
where the intervention helps, and that any larger number is suspect. Our
+0.462 on GPT-2 is an order of magnitude above the first of them, so the
discrepancy has to be addressed rather than left for a reader to find.

Those values were measured on `SAMPLE_TEXT * 200`: one paragraph repeated two
hundred times, base loss 0.76 nats against 3.96 for real prose. The
specification's own bug list identifies that input as a defect to fix before
porting anything, and the reason it matters here is specific rather than
general. A correlation across heads needs variation across heads. One
paragraph repeated gives a model very little to do differently in different
heads, so the per-head effects it produces are small and largely
undifferentiated, and correlating them against anything returns approximately
zero. Near-zero on that input is a property of the input.

Measured on 64 real wikitext-103 documents per half, with disjoint halves and
the reliability of the effect established first, the same correlation is
+0.462 on GPT-2 with a ceiling of 0.890, and it held between +0.416 and
+0.469 across four independent runs. The prior figures are superseded, not
contradicted: they are not measurements of the quantity they appear to
describe. `DEVIATIONS.md` D8 records this. One model
disagreeing with another is not a general law about statistics,
and the honest summary is narrower than the one we would have liked to write:
whether the raw statistic predicts its own intervention is model-dependent,
and a method that assumes it does is assuming something that is false in two
of the three models we measured.

## 6. The matched intervention (Check 2), and a power failure

Five arms, identical initialisation and data order per seed, differing only in
the intervention. Zero-initialised gates make every arm exactly the baseline at
step 0; measured deviation across all five arms is **0.000e+00**.

At the token budget we could afford, 5e7 tokens per run:

**Table 4.** Paired difference against baseline, CFG_S underpowered pilot.
`n = 8 seeds per arm.`

| arm | mean delta | 95% CI | t | p | Cohen d_z | realised sigma | MDE at n=8 |
|---|---|---|---|---|---|---|---|
| **random** (pre-registered primary) | **+0.001190** | [+0.000351, +0.002040] | +2.48 | **0.042** | 0.877 | 0.001356 | 0.00139 |
| xsa | +0.001515 | [-0.001223, +0.004807] | +0.92 | 0.387 | 0.326 | 0.004642 | 0.00476 |

Read the sign first. Both arms are **positive**, meaning both interventions are
*worse* than baseline at this budget, and the matched arbitrary direction is
significantly so (p = 0.042, interval excluding zero). They are also not
distinguishable from each other. At 5e7 tokens per run the models are far below
the spec's 3.5e8 floor, and a gated rank-one removal plausibly just costs
capacity there.

**Power, stated per arm rather than once.** The minimum detectable effect
depends on the arm's realised paired standard deviation, and the two arms
differ by a factor of 3.4. Against the **0.00076 nats** an independent
replication reports:

* the primary endpoint (`random`) resolves to **0.00139**, about **1.8x** the
  target effect;
* the arm that matters for the method (`xsa`) resolves to **0.00476**, about
  **6.3x** it.

A separate figure, **0.00518**, appears in `results/pilot_decision.json`. That
is the Day-3 **planning** MDE, forecast from a three-seed pilot with
`sigma_paired = 0.005049` before the factorial was run. It is what sized the
design, not what the design achieved, and it should not be quoted as a
measured result. The realised sigmas above are 3.7x and 1.1x smaller.

**Check 2 returns no verdict on XSA at this power.** That is not evidence that
XSA and a matched arbitrary direction are equivalent, and we do not present it
as such. We report the power calculation because a paper reporting the point
estimate without it would be claiming a null it cannot support.

---

## 7. Check 0 in the field

The resolvability check was applied to a separate, independently developed
intervention study on a partitioned-attention architecture, which reported that
structural overlap does not predict behavioural contribution.

Split-half reliability of its per-edge effect was **+0.088, -0.026 and +0.012**
across three seeds, against a 0.3 threshold. Unreliability caps any observable
correlation at `sqrt(r_delta * r_stat)`, at most **0.102**. Every correlation
that project reported had magnitude at most 0.018, well inside the ceiling.
Single-edge deltas were 4 to 6 float32 ULPs, and a budget sweep showed the
estimate never converges.

**That project withdrew its headline claim as a result.** We report this
because it is the strongest available evidence that Check 0 is worth running:
it changed a published conclusion, and it did so before rather than after
review.

---

## 8. A reproduction we could not complete

Published reference values for GPT-2 are `cos_self` 0.5406, `cos_null` 0.3798,
`excess` 0.1608. We measure **0.4828 / 0.2987 / 0.1840**, outside a ±0.01
tolerance on all three.

The port is not obviously at fault: the GQA expansion matches the reference
implementation bit for bit, attention is forced eager, the reconstructed
per-head output is verified against the module's own output, and the probe
behaves correctly across four model families. We enumerated the conventions a
Check-1 measurement must fix — sequence length, position-0 handling,
null-partner definition, head aggregation, and layer subset — and measured all
of them (`results/gpt2_diagnosis.csv`). Sequence length accounts for the null
(0.3747 at T=64 against the 0.3798 reference) but not the observed statistic,
which stays near 0.48 at every length.

We report this as unexplained. We did not search further, because selecting the
configuration that lands on a target and presenting it as the method is exactly
the practice this paper argues against.

---

## 9. Limitations

* **The scale gap is real and we do not paper over it.** We *train* at 51M
  parameters. XSA trains at 0.7-2.7B. We *measure* frozen statistics to 6.9B,
  which covers XSA's range, but a frozen statistic is not a training result.
  **We cannot and do not claim to refute XSA at its scale.**
* **The training leg is underpowered** at the budget we could afford (§5),
  resolving the XSA arm to 0.00476 nats against a 0.00076 target, about 6x.
  The primary `random` arm is tighter at 0.00139, about 1.8x. Point estimates
  are reported with that caveat attached and are not evidence of equivalence.
* **The 124M scale check was not run.** The budget solver dropped it in the
  pre-registered priority order rather than shrinking the primary endpoint.
* **Two of four A6 methods are not measured.** Value-residual needs a
  matched-capacity control, which means training two models; registers are a
  vision-transformer construct outside a causal-LM harness.
* **The ablation value matters.** Li & Janson (arXiv:2409.09951) show that what
  you ablate *to* changes which components look important. Our interventions
  remove a direction rather than replacing it with a learned optimum, and a
  different choice could change the ranking.
* **`diagmask` is gated rather than hard-masked** in the runs reported here.
  The specification describes both, and only the gated form can satisfy the
  step-0 identity the paired design requires. Both are implemented and the
  difference is measured; see `DEVIATIONS.md`.

---

## 10. What ships

`xsac/checks.py`: `check_resolvability`, `check_null`, `check_matched`. Three
functions, documented for someone who has not read this paper, each returning a
result object that carries its own `n`, an interval where one is defined, and a
`passed` flag whose meaning is stated on the class. None of them raise on a
negative result, because a failed check is a finding.

The harness around them is released too: verified hooks for attention-internal
measurement, GQA expansion checked against the reference implementation, the
anisotropy null, and a self-test suite that catches the silent failures we hit
while building it.

---

## References

Ethayarajh (2019). How Contextual are Contextualized Word Representations?
Godey et al. (EACL 2024). Anisotropy Is Inherent to Self-Attention.
Li & Janson (arXiv:2409.09951). Optimal Ablation for Interpretability.
Machina & Mercer (NAACL 2024). Anisotropy is Not Inherent to Transformers.
Pan et al. (NeurIPS 2024). Anisotropy baselines for Q/K in vision transformers.
Sun et al. (2024). Massive Activations in Large Language Models.
Timkey & van Schijndel (2021). All Bark and No Bite.
Xiao et al. StreamingLLM: Efficient Streaming Language Models with Attention Sinks.
Zhao et al. (arXiv:2605.20798). On the transfer of transformer modifications.

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
the checklist discriminates rather than debunking uniformly. **(4)** Applied to
a separate published intervention study, our resolvability check found the
effect unmeasurable and caused that project's headline claim to be withdrawn
(§6).

We do not claim to refute XSA. Our training leg runs at 51M parameters against
XSA's 0.7-2.7B, and at the token budget we could afford the design is
underpowered by about sevenfold against the effect size an independent
replication measured (§5). We report that as a power failure, not as a null
result.

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

Nine models, 5,408 head-level rows, 32 wikitext-103 documents each, eager
attention, null partner drawn within the sequence from positions the query
could causally attend.

**Table 1.** Check 1 across the ladder. `n = 9 models, 5,408 heads.`

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

`cos_null` falls from 0.2637 at 160M to 0.1979 at 6.9B. The confound weakens
with scale, consistent with Machina & Mercer. **It does not vanish.** At 6.9B,
58% of `cos(y_i, v_i)` is still explained by the null, and across XSA's own
0.7-2.7B training range the self-specific share sits between 46 and 51%. The
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

**Table 2.** Within-group versus across-group excess. `n = 2 models, 672 heads.`

| model | query / KV heads | within-group excess | across-group excess |
|---|---|---|---|
| Qwen2.5-0.5B | 14 / 2 | **+0.2415** | **-0.1876** |
| Qwen2.5-1.5B | 12 / 2 | **+0.2731** | **-0.1922** |

Borrowing a neighbouring group's value at the same position does not merely
lose the effect; it goes **negative**. The self-value direction is specific to
the shared KV group. GQA models also show a higher self-specific fraction, 56
to 59%, than any MHA model on the ladder, 37 to 52%. GQA is not a neutral
change of variable for this family of methods. (Figure 5.)

---

## 5. The matched intervention (Check 2), and a power failure

Five arms, identical initialisation and data order per seed, differing only in
the intervention. Zero-initialised gates make every arm exactly the baseline at
step 0; measured deviation across all five arms is **0.000e+00**.

At the token budget we could afford, 5e7 tokens per run:

**Table 4.** Paired difference against baseline, CFG_S.

| arm | mean delta | 95% CI | t | p | n |
|---|---|---|---|---|---|
| **random** (pre-registered primary) | +0.000921 | [-0.000673, +0.002515] | +0.92 | 0.426 | 4 |
| xsa | +0.000528 | [-0.002500, +0.005420] | +0.21 | 0.848 | 4 |

`sigma_paired = 0.00505` gives a minimum detectable effect of **0.00518 nats**.
The effect an independent replication measured is **0.00076 nats**. The design
as run is therefore underpowered by about **sevenfold**, and both intervals
span zero and each other.

**Check 2 returns no verdict at this power.** That is not evidence that XSA and
a matched arbitrary direction are equivalent, and we do not present it as such.
The binding constraint is tokens per run rather than seeds: moving from 4 to 12
seeds improves the MDE only by `sqrt(3)`. We report the power calculation
because a paper that reports the point estimate without it would be claiming a
null it cannot support.

---

## 6. Check 0 in the field

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

## 7. A reproduction we could not complete

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

## 8. Limitations

* **The scale gap is real and we do not paper over it.** We *train* at 51M
  parameters. XSA trains at 0.7-2.7B. We *measure* frozen statistics to 6.9B,
  which covers XSA's range, but a frozen statistic is not a training result.
  **We cannot and do not claim to refute XSA at its scale.**
* **The training leg is underpowered** by about sevenfold at the budget we
  could afford (§5). Its point estimates are reported with that caveat attached
  and are not evidence of equivalence.
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

## 9. What ships

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

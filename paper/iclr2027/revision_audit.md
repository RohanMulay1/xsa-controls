# Manuscript revision and audit

Date: 6 September 2026. Status: **edited and compiled, with disclosed research limitations**. This is not an unconditional submission-readiness or plagiarism certification.

## Scope and framework use

Revised `main.tex` and rebuilt `out/main.pdf`. The user authorised corrections where the draft contradicted archived results or implementation. No experiments were rerun, no new inferential tests were added, and no research code, stored results, bibliography entries, or figure artwork were changed.

The [LaTeX paper framework](https://github.com/witold-andelie/claude-latex-paper-skill) informed the claim-to-evidence review, direct technical prose, citation checks, and compile/visual verification. The [academic writing framework](https://github.com/WenyuChiou/academic-writing-skills) informed the manuscript contract, source authority, semantic locks, terminology profile, exact-candidate prose audit, and explicit record of unresolved issues. These were applied as revision frameworks, not as prose templates. Their relevant instructions were read, and their verification/audit scripts were inspected before execution. Nothing was installed into the user's skill configuration.

Authority was: archived result files for measurements; implementation for operational definitions; primary publications for external attribution. Historical summaries and the manuscript did not override those sources. The contract and five remaining research/provenance issues are recorded in [manuscript_state.json](manuscript_state.json).

## Evidence-backed corrections

| Topic | Correction and evidence |
| --- | --- |
| Main training result | Negative arm-minus-baseline loss means improvement. XSA lowers loss by 0.002924 nats, not raises it. Its archived Holm-corrected p is 0.0023. The random arm's +0.001056, p=0.197 is a nondetection, not equivalence. Checked `results/paired_tests_s.csv`, `results/factorial_s.csv`, and `xsac/stats.py`; independently reproduced both mean contrasts from seed-level losses. No direct XSA-versus-random significance claim was added. |
| Scope of redundancy claim | A benefit from learning gates during training does not establish that the component is dispensable in a frozen model. Cosine association is not predictive validation or a causal explanation. Abstract, discussion, and conclusion now maintain those distinctions. |
| Random direction and gate semantics | `xsac/arms.py` fixes a Gaussian-normalised direction per layer/head, including across seeds. It is not resampled per position. Gate/rank-one structure is shared, but removed norms are not matched. Training gates are zero-initialised, learned, and applied at all positions; frozen interventions use multiplier one, one head at a time, and omit the first token. Negative gates add the projection. Stabilisation constants are disclosed alongside the preserved idealised equation. |
| Projection and null | Projecting the full output onto its own value also removes aligned earlier-token contributions; this is not diagonal attention deletion. `xsac/frozen.py` samples the null from strictly earlier positions, excluding the current token and omitting the first query. |
| Reliability | `xsac/stats.py` and `results/reliability.csv` use raw split-half Spearman reliability, not a Spearman–Brown correction. Behavioural reliabilities are 0.795, 0.419, and 0.531; structural reliabilities are all high. Thresholds apply to behavioural reliability, not the geometric mean. Only three models have this reliability evaluation. |
| Pooled correlations | Archived disattenuation divides pooled-statistic correlations by a diagnostic constructed from half-sample reliabilities. Values and equation are retained but not called calibrated latent pooled correlations. Raw correlations lead the interpretation; all six were reproduced from `results/a2_per_head.csv`. Squared Spearman correlation is not reported as explained variance. |
| Existing scatter and earlier run | `xsac/figures.py` plots half-A self-cosine against pooled effects, while annotations use pooled statistics. The caption now says so; artwork is unchanged. The earlier Pythia excess correlation 0.487 was raw, not adjusted (`results/a2_budget_comparison/a2_correlations_n64_halfA_pairing.csv`). Run changes prevent attributing its difference from 0.189 solely to reliability. |
| Ladder and GQA | Nine MHA models span 124M to 6.9B; three GQA models bring the total to twelve. Reliability-sample lengths are not imposed on every ladder run. `results/ladder.csv` supports 0.483 to 0.340 raw self-cosine, not the original 0.44 start. `results/gqa.csv` and `gqa_within_across` support an own-group/next-group same-position contrast against an own-group earlier-position null. Negative excess need not mean negative raw cosine, and TinyLlama does not match Qwen's magnitude. |
| Generality examples | Both attention sinks and massive activations survive their respective controls. For activations, 12.87186 minus 3.64521 leaves 9.22665, about 71.68%. `results/generality.csv` and `scripts/run_generality.py` use different, explicitly described nulls for the two examples. |
| Reconstruction gate | Historical perturbation notes are distinguished from current archived reconstruction errors (roughly 0.00163–0.00167 under a 0.01 threshold). Tests cover selected first/middle/last layers and one document per model, not all possible layers or semantic correctness. The unsupported fourth perturbation claim was removed. |
| Gate plot and extra arms | The gate figure contains endpoints, not trajectories. Positive XSA layer means do not show every head stayed positive throughout training. Later random-arm layer means are negative. Implemented mean-value and diagonal-mask variants are described from `xsac/arms.py`/`xsac/intervene.py`, without claiming they were evaluated in the primary factorial. |
| Design, power, and budget | The archived factorial has 24 completed cells, eight seeds, and 399,900,672 tokens per run. Architecture and data description were checked against `xsac/config.py` and `data/prepare.py`. MDE is an approximation, and extrapolated power curves are not additional runs. `BUDGET.md` distinguishes $15.58 billed spend, a $20 ceiling, and 16.98 summed cell-hours from billed uptime. $18.16 and $13.41 are historical forecasts, not realised spend. |
| Reproducibility and registration | Hand-maintained manuscript tables are not automatically generated by CI. Stale manifest entries do not certify them. No separate timestamped registration was located, so the manuscript says protocol-specified. The failed archived GPT-2 target reproduction in `results/gpt2_target_check.json` is now disclosed in Limitations. |

Original citation keys and bibliography bytes were preserved. A linked footnote now attributes XSA to its [original method paper](https://arxiv.org/abs/2603.09078). The external -0.00076 figure reference was checked against the first H200 table in [PR 264](https://github.com/KellerJordan/modded-nanogpt/pull/264): baseline loss 3.27941 at 1440 steps versus XSA 3.27865 at 1410 steps, ten runs each. These are not matched-step replication targets. The exact configuration/source for the separate -0.017 line was not established; its existing artwork is retained and the caption explicitly flags that limitation.

## Prose and originality review

The revision removed stock rhetorical phrases, unsupported absolutes, artificial transitions, repeated argumentative scaffolding, inflated wording, unnecessary hedges, and em dashes. Technical terms and necessary qualifications were retained. The prose projection used for the audit decreased from approximately 3,931 to 3,602 words; this is a manuscript-specific count with mathematical placeholders, not a conventional publication word count.

The final extracted prose has zero findings from the configured candidate-pattern audit; the raw-source semantic/profile audit also has zero findings. Both source and rendered PDF contain zero em dashes. These checks support an editorial review, not a claim that an algorithm can determine human authorship.

A bounded exact-overlap check found **no matching 12-word sequences** between the final manuscript's normalised prose and nine successfully retrieved pages: the two framework skill files, their style/prose references, the XSA paper HTML, the Voita and Ethayarajh ACL pages, the transformer-circuits article, and PR 264. Only public source pages were downloaded; manuscript text was not uploaded to a similarity service. This check does not cover all literature, detect every paraphrase, or certify original research. Standard technical expressions were not distorted merely to avoid overlap.

Citation claims were reviewed against primary sources: [attention](https://arxiv.org/abs/1706.03762), [head pruning](https://arxiv.org/abs/1905.10650), [specialised heads](https://aclanthology.org/P19-1580/), [QK/OV circuits](https://transformer-circuits.pub/2021/framework/index.html), [anisotropy](https://aclanthology.org/D19-1006/), [GQA](https://arxiv.org/abs/2305.13245), [Pythia](https://arxiv.org/abs/2304.01373), and the [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Access to the [Holm source](https://www.jstor.org/stable/4615733) was metadata-only; the implemented correction was checked locally, without claiming a full-text review. Literature claims were narrowed to the settings actually supported.

## Verification of the delivered candidate

- [revision_checks.json](revision_checks.json): **63/63 integrity and arithmetic checks pass**. Preserves all three displayed equation bodies, numerical table content, labels, original citation-key set, bibliography bytes, and author comments. Reconciles reported table values with archives and reproduces seed means and pooled raw correlations.
- `python paper/iclr2026/build.py --preview`: success, seven-page PDF. All 16 geometry checks and all 10 float-placement checks pass. No undefined references/citations or overfull boxes.
- Every page was visually inspected for clipping, columns, equations, tables, figures, captions, and references. The final abstract adjustment was rechecked on the exact final page; the remaining pages were also rechecked.
- Upstream `verify_paper.py`: exit 0, with 15 soft bibliography warnings. All twelve existing entries lack locator fields, and three existing entries are uncited: `efron1979bootstrap`, `hoffmann2022training`, `spearman1904proof`. They were intentionally left intact.
- Accepted build warnings: two underfull vertical boxes and three inherited bibliography underfull horizontal boxes. A Fontconfig configuration warning did not prevent font rendering. Visual inspection found no corresponding clipping or unreadable text.
- `git diff --check`: passes. Research files, result archives, bibliography, and artwork remain unchanged. Only manuscript source and revision/audit sidecars are changed or added.

The reusable local audit is [audit_revision.py](audit_revision.py). It runs without remote scripts for integrity checks. To reproduce the additional framework checks, supply `--framework-dir` pointing to the four inspected upstream scripts; exact script hashes are in `revision_checks.json`. This run used `C:/Users/rohan/AppData/Local/Temp/xsa-writing-audit-20260906`. Temporary scripts are not a permanent project dependency.

## Remaining research and provenance limitations

1. The exact external -0.017 configuration/source remains unresolved.
2. The archived GPT-2 target reproduction failed; this revision does not repair or rerun it.
3. No separately timestamped registration was located.
4. Half-sample reliabilities do not by themselves calibrate the pooled-correlation disattenuation.
5. Existing scatter coordinates and correlation annotations use different aggregation conventions, now disclosed.

These limitations are visible in the manuscript or its captions and tracked in the state file. Resolving them would require source clarification, methodological changes, or new research work beyond this writing revision.

## Initial revision artifact identity

Baseline commit: `5bf97be60b3336cee5d75ec51e242e762e861674`.

| Artifact | SHA-256 |
| --- | --- |
| Baseline main.tex | `ada1d5c2f1cf1d224851daaa92fecab6858f4f973719b2f700c676ec9ce44e4f` |
| Baseline main.pdf | `24f3540e80c1c4b3789e2d8f624b5c3d3fccce77e68a89be264037e705f47600` |
| Final main.tex | `89348842d43c97af0445f354c69080d473aa4799b4a55f15454d4a8624014ec1` |
| Final main.pdf | `6e166034d2b9e61757d1c85dd5d44b888d373637cfd6bf3a33a5c7e677f3c74f` |
| Unchanged bibliography | `2e0e26f9dbcddb21fc5fcc82e6cd508356a0c6772d60f7b5a4086d14762ca438` |

## Follow-up: explicit objective and research conclusion

At the user's request, retained the existing research title, added the run-in heading "Research objective." to the existing objective paragraph, and renamed Section 7 "Research Conclusion". No findings or scientific prose were changed. Rebuilt the seven-page PDF; geometry and float checks pass. Rechecked the title/objective and conclusion pages visually. All 63 integrity checks still pass, with no configured prose flags or rendered em dashes. The build retains two underfull vertical-box warnings and three bibliography underfull horizontal-box warnings. The overlap check above belongs to the initial revision and was not rerun for these heading-only changes.

Current source SHA-256: `116bb151bb21f0224c978a3c46310af96770480c0a920a0e1fa73115ce367ae9`.

Current PDF SHA-256: `439817f397d68a66142370c1fcb0ae5bf3c684cec25d0695ef34068adaa6989e`.

The machine-readable check report and state file identify this follow-up candidate.

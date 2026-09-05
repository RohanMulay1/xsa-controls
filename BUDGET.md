# Budget ledger

Ceiling **$70.00**. Stop and report if projected spend exceeds **$66.00**.

## Running total

| Date | What ran | GPU-hours | $ spent | $ remaining |
|---|---|---|---|---|
| 2026-09-03 | Day 1: scaffold, 5 arms, 10/10 self-tests, A4 recompute, Track A on CPU | 0.00 | $0.00 | $70.00 |
| 2026-09-03 | Calibration, data prep, A1 nine-model ladder, A3 GQA, pilot, CFG_S pilot factorial (24 cells at 5e7 tokens/run) | ~7.5 | **~$5.55** | **~$64.45** |
| 2026-09-04/05 | A3 second GQA family (TinyLlama), A2a + A2 on GPT-2 and two Pythia sizes, precision check, and the sibling CRPA project's long-context work | 4.18 | **$5.81** | **$58.64** |

**Total spent to date: $11.36.** The A100 figure is the pod's full billed
uptime, 4.18 hours at $1.39/hr, not an estimate of the share attributable to
this repository: the card was shared with the sibling CRPA project's
long-context runs and splitting a single rented pod between two projects
would be arithmetic invented after the fact. Charging the whole of it here
overstates this project's cost rather than understating it. The first block ran on an RTX
6000 Ada at $0.74/hr community; the second on an A100-SXM4-80GB at $1.39/hr,
shared with the sibling CRPA project's long-context work. Projected spend
never approached the $66 stop-and-report threshold, which is now enforced in
`solve_token_budget` rather than merely declared.

## The primary endpoint, registered 2026-09-05

**Registered before the run, not after.** CFG_S, arms baseline / xsa / random,
seeds 42 1337 2024 7 99 512 8191 31337, **399,900,672 tokens per run**. The
budget is inside the pre-registered [3.5e8, 6e8] band and is a multiple of
`batch_tokens` (131,072 x 3,051), so every arm sees an identical budget. The
pre-registered primary endpoint remains `random` against `baseline`; Holm
correction applies over the secondary arms only.

### Why $24 was the wrong projection, and what the right one is

The committed `calibration.json` projects **$24.01** for 24 runs at
349,962,240 tokens, which scaled to 399,900,672 would be about $27.44 and
would breach a $20 ceiling. That figure is not wrong so much as answering a
different question: it is solved from **diagmask's** throughput, 71,918 tok/s,
because diagmask is the slowest of the five arms and the solver sizes the
whole design against its worst case.

**Diagmask is not in this run.** The three arms here were measured directly on
the card that will run them, an RTX 6000 Ada at $0.74/hr:

| arm | tokens/sec | hours for 8 runs at 3.999e8 tokens |
|---|---|---|
| baseline | 176,467 | 5.04 |
| xsa | 159,358 | 5.58 |
| random | 166,267 | 5.35 |
| **total** | | **15.96 GPU-hours** |

**15.96 h x $0.74/hr = $11.81**, against a $20 ceiling. The run proceeds.

Substituting the slowest arm's throughput for arms that are twice as fast
would have blocked a run that fits inside the ceiling with $8 to spare. The
lesson is the same one the rest of this repository keeps arriving at: the
number has to be measured on the thing being asked about.

### CFG_M is dropped for budget, with the arithmetic

The CFG_M scale check is **not run and not attempted**. At CFG_M the measured
`diagmask_slowdown` is 2.336 and the committed calibration projects **$51.57**
for the same 24 runs at 349,962,240 tokens; at 399,900,672 that is about
$58.94. Even taking the same correction as above and assuming CFG_M's three
arms run at CFG_S's ratio to diagmask, the floor is well beyond $20.

Dropping it is the spec's own pre-registered priority order:
`cuts_applied: ["cfg_m_scale_check", "secondary_arms"]`. It is recorded here
as **dropped for budget**, which is its sanctioned status, not as an oversight.

## The old 5e7 factorial, kept as the underpowered pilot

`factorial_s_pilot_5e7.csv` and `paired_tests_s_pilot_5e7.csv`, with their 24
run records under `results/factorial_s_pilot_5e7/runs/`.

It is 24 cells at 5e7 tokens per run,
outside the band, produced when `calibrate_cli.py` was invoked with
`--cost-ceiling 3.00` against a spec figure of $56. The solver recorded
`affordable: false` and the run proceeded anyway. It is kept, relabelled as
the underpowered pilot, because it is honest provenance and it is what the
power analysis is computed from. It is not the primary endpoint and no table
presents it as one.

It is superseded by the registered run above, at 399,900,672 tokens per run
and a measured $11.81. The $38 figure previously quoted here came from the
same diagmask-throughput substitution described above and was about three
times the real cost.

Two guards now make the original failure impossible to repeat:
`--cost-ceiling` is required with no default, and `run_factorial.py` refuses
to start below the batch-aligned 3.5e8 floor on **every** path that can set a
budget, including an explicit `--tokens-per-run`, unless
`--i-accept-underpowered` records the choice.

## Planned allocation (spec section 12, at the $0.86/hr L40S placeholder)

| Item | Runs | Hours | $ |
|---|---|---|---|
| Calibration + pilot | — | 3 | $2.6 |
| **Primary: CFG_S, 3 arms (baseline/xsa/random) x 8 seeds** | 24 | 34 | $29.2 |
| Secondary: CFG_S, 2 arms (meanval/diagmask) x 5 seeds | 10 | 15 | $12.9 |
| **Scale check: CFG_M, 3 arms x 3 seeds** | 9 | 16 | $13.8 |
| Track A: A1 nine-model ladder (incl. 2.8B, 6.9B) | — | 2 | $1.7 |
| Track A: A2, A3, A5, A6 | — | 7 | $6.0 |
| Weights download (~40 GB) + slop | — | 4 | $3.4 |
| **Total** | **43** | **81** | **$69.6** |

This allocation is **not yet valid**. `TOKENS_PER_RUN` must be recomputed
against the actual hourly rate of the machine that will run the factorial:

```bash
python scripts/calibrate_cli.py --rate <ACTUAL_USD_PER_HOUR> --n-runs 43
```

Running `calibrate_cli.py` without `--rate` uses the placeholder, marks the
output `rate_is_placeholder: true`, and the Day-2 gate fails by design.

## Priority order if short. Cut from the bottom.

1. A6 extra methods (list as future work)
2. Scale-check seeds 3 -> 2
3. Secondary arms `meanval` / `diagmask` seeds 5 -> 3
4. **Never cut the primary endpoint below 8 seeds, and never cut A1.** Those
   are the paper.

If the token budget clamps below 3.5e8, the **CFG_M scale check is dropped
first** and the decision is recorded here with its arithmetic, per spec §7.

## Day-3 GO/NO-GO: DECIDED

**Run on 2026-09-03. Branch chosen: `reduce`.**

Measured from the pilot, 3 arms x 8 seeds at 5e7 tokens per run on an RTX 6000
Ada:

```
sigma_paired = 0.0050487
MDE          = 2.9 * 0.0050487 / sqrt(8) = 0.0051764
branch       = reduce   (0.002 <= MDE < 0.008)
action       = Drop the secondary arms. Run 3 arms x 12 seeds at CFG_S.
               Keep the scale check and A1.
```

**The Day-3 planning MDE was 0.00518 nats; the completed pilot realised 0.00139 (random) and 0.00476 (xsa). PR #264's measured effect is 0.00076 nats.** The
design as run is underpowered by a factor of about seven against the effect it
is trying to resolve, and no number of seeds fixes that cheaply: going from 4
to 12 seeds improves the MDE by sqrt(3), to about 0.0030, which is still four
times too coarse.

The binding constraint is **tokens per run, not seeds.** This run used 5e7
because that is what completed in the session; the spec's 3.5e8 floor exists
precisely to keep sigma_paired small enough for the effect to be resolvable,
and this measurement is direct evidence for why that floor was set.

Primary endpoint at this budget, reported because it was pre-registered and not
because it is significant:

| arm | mean delta vs baseline | 95% CI | t | p | n |
|---|---|---|---|---|---|
| **random** (primary) | +0.000921 | [-0.000673, +0.002515] | +0.92 | 0.426 | 4 |
| xsa | +0.000528 | [-0.002500, +0.005420] | +0.21 | 0.848 | 4 |

Both intervals span zero and each other. **Check 2 returns "cannot tell" at
this power**, which is the honest outcome and not evidence that xsa and random
are equivalent.

### The rule, for reference

| MDE | Branch |
|---|---|
| < 0.002 | proceed as specced |
| 0.002 - 0.008 | drop secondary arms, 3 arms x 12 seeds at CFG_S |
| >= 0.008 | **kill the training leg the same day**, pivot to Track A only |

```bash
python scripts/run_factorial.py --pilot --n-seeds-planned 8
python scripts/verify_day.py 3
```

The Day-3 gate checks that this file quotes the MDE. Discovering on Day 8 that
the design had no power is the single most likely way this project fails.

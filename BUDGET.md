# Budget ledger

Ceiling **$70.00**. Stop and report if projected spend exceeds **$66.00**.

## Running total

| Date | What ran | GPU-hours | $ spent | $ remaining |
|---|---|---|---|---|
| 2026-09-03 | Day 1: scaffold, 5 arms, 10/10 self-tests, A4 recompute, Track A on CPU | 0.00 | $0.00 | $70.00 |
| 2026-09-03 | Calibration, data prep, A1 nine-model ladder, A3 GQA, pilot, CFG_S factorial (12 of 24 cells) | ~7.5 | **~$5.55** | **~$64.45** |

**Total spent to date: approximately $5.55** on an RTX 6000 Ada at $0.74/hr
community. Projected spend never approached the $66 stop-and-report threshold,
which is now enforced in `solve_token_budget` rather than merely declared.

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

Measured from the pilot, 3 arms x 4 seeds at 5e7 tokens per run on an RTX 6000
Ada:

```
sigma_paired = 0.0050487
MDE          = 2.9 * 0.0050487 / sqrt(8) = 0.0051764
branch       = reduce   (0.002 <= MDE < 0.008)
action       = Drop the secondary arms. Run 3 arms x 12 seeds at CFG_S.
               Keep the scale check and A1.
```

**The MDE is 0.00518 nats. PR #264's measured effect is 0.00076 nats.** The
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

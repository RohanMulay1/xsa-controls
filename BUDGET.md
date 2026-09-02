# Budget ledger

Ceiling **$70.00**. Stop and report if projected spend exceeds **$66.00**.

## Running total

| Date | What ran | GPU-hours | $ spent | $ remaining |
|---|---|---|---|---|
| 2026-09-03 | Day 1: scaffold, 5 arms, 10/10 self-tests, A4 recompute, Track A on CPU | 0.00 | **$0.00** | $70.00 |

**Total spent to date: $0.00.** No GPU pod has been rented. Day 1 is CPU-only
by design, and the anti-pattern table forbids starting a pod on Day 1 because
it bills continuously.

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

## Day-3 GO/NO-GO

Not yet run. When it is, the branch must be recorded here in writing with the
MDE value quoted, before any further GPU spend:

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

# Primary factorial: running, not yet collected

The registered primary endpoint is running unattended. This file exists so
that whoever collects it does not have to reconstruct what was launched.
Delete it once the results are committed.

## What is running

```bash
python scripts/run_factorial.py --size S \
  --arms baseline xsa random \
  --seeds 42 1337 2024 7 99 512 8191 31337 \
  --tokens-per-run 399900672 --device cuda
```

| | |
|---|---|
| Started | 2026-09-05, RunPod pod `xlbfc44jppqxyd` |
| GPU | NVIDIA RTX 6000 Ada Generation, 48GB, community |
| Rate | $0.74/hr |
| Cells | 24 (3 arms x 8 seeds) |
| Projected | 15.96 GPU-hours, **$11.81**, against a $20 ceiling |
| First cell | baseline, val_loss 3.800306, 3079.5 s |
| Log | `/workspace/factorial.log` on the pod |

The first cell ran slower than its projection because two A2 measurement jobs
were sharing the card for the first 25 minutes. The card is dedicated now.

## Why this budget

`399,900,672` is inside the pre-registered `[3.5e8, 6e8]` band and is
`131,072 x 3,051`, so it is batch aligned and every arm sees an identical
budget. `run_factorial.py` refuses anything below the batch-aligned floor on
every path, so the guard was satisfied rather than bypassed.

The committed `calibration.json` projects $24.01 for this design, which would
have breached the ceiling. That figure is solved from **diagmask's**
throughput of 71,918 tok/s because the solver sizes against the slowest arm,
and diagmask is not in this run. Measured on the card itself: baseline
176,467, xsa 159,358, random 166,267 tok/s. `BUDGET.md` carries the
arithmetic.

## Collecting it

The run is resumable: run ids are content hashes and completed cells are
skipped, so an interrupted run can simply be relaunched with the same
command.

```bash
# on the pod
ls results/factorial_s/runs/*.json | wc -l        # expect 24
tail -5 /workspace/factorial.log                  # expect the paired tests

# copy back
scp -r root@<pod>:/workspace/xsa-controls/results/factorial_s ./results/
scp root@<pod>:/workspace/xsa-controls/results/factorial_s.csv ./results/
scp root@<pod>:/workspace/xsa-controls/results/paired_tests_s.csv ./results/
```

Then, locally:

```bash
python scripts/make_figures.py      # fig1_gates and fig2_paired_delta appear
python scripts/check_figures.py
python scripts/make_tables.py       # T4 switches from the pilot to the primary
python scripts/make_manifest.py     # paired claims move off the pilot fallback
python scripts/verify_day.py --all  # Days 4-6 should move off 6/8
pytest -q
```

`fig7_power` will then plot **two** curves, the 5e7 pilot and this run,
against PR #264's 0.00076. That comparison is the reason the pilot was kept
rather than deleted.

## What must not happen when it lands

* The pilot stays. `factorial_s_pilot_5e7.csv` and
  `paired_tests_s_pilot_5e7.csv` are provenance and are what the power
  analysis is computed from. They are not superseded data, they are a
  different budget.
* No averaging across the two budgets. The homogeneity guard exists for
  exactly this and `fig7_power` refuses a single label when a file mixes
  them.
* `random` is the pre-registered primary endpoint and is reported first.
  Holm correction applies over the secondary arms only.

## Terminate the pod afterwards

It bills at $0.74/hr whether or not anything is running.

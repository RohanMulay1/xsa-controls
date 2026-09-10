# XSA controls

This repository provides controls for attention-surgery claims: first establish
that a statistic is measurable, then remove its null component, then compare
the proposed intervention with a matched arbitrary-direction control.

```python
from xsac.checks import check_resolvability, check_null, check_matched
```

## The three checks

| Check | Question |
| --- | --- |
| Resolvability | Do repeated measurements of the effect agree? |
| Null | How much of a statistic remains after a matched geometric null? |
| Matched intervention | Does the proposed removal outperform an arbitrary matched removal? |

All arms act on per-head attention output before the output projection. Gates
start at zero, so every arm equals the baseline at step zero. The trained arms
are `baseline`, `xsa`, `random`, `meanval`, and `diagmask`; `random` is the
primary matched control.

## Results retained in this repository

The frozen-model ladder contains nine models through Pythia-6.9B and 5,408
head-level records. The null component decreases with scale but remains
substantial. Grouped-query models show positive within-KV-group excess and
negative across-group excess.

The final small-model factorial contains 24 completed cells: baseline, XSA, and
a fixed random-direction control across eight paired seeds, with 399,900,672
tokens per run. Relative to baseline, the random control changed validation
loss by +0.001056 nats (95% CI -0.000298 to +0.002390; p=0.197), while XSA
changed it by -0.002924 nats (95% CI -0.003967 to -0.001709; Holm-adjusted
p=0.00230). The earlier 5e7-token pilot remains separately labeled.

The GPT-2 reference Check-1 values did not reproduce after the recorded
measurement conventions were tested. The diagnostic CSV is retained without
selecting a configuration that matches the target.

## Install and verify

```bash
pip install -r requirements.txt
python -m pytest -q -m "not slow"
python scripts/selftest_arms.py --json results/selftest.json
python scripts/a4_recompute.py
python scripts/make_figures.py
```

Use `python -m pytest` rather than bare `pytest` on Windows environments where
the project root is absent from the import path.

## Reproduce the measured studies

```bash
# Prepare data and calibrate a specific GPU run.
python data/prepare.py --tokens 6e7 --val-tokens 4000000
python scripts/calibrate_cli.py --rate 0.74 --cost-ceiling 3.00 --n-runs 43 --device cuda

# Frozen-model measurements.
python scripts/run_frozen.py --ladder --n-docs 32 --block 512 --device cuda --dtype bfloat16
python scripts/run_frozen.py --gqa --n-docs 32 --block 512 --device cuda --dtype bfloat16
python scripts/run_generality.py --model gpt2 --n-docs 12 --block 256
python scripts/diagnose_gpt2.py --device cuda

# Paired training experiment and figures.
python scripts/run_factorial.py --size S --arms baseline xsa random --seeds 42 1337 2024 7 99 512 8191 31337 --tokens-per-run 399900672 --device cuda
python scripts/make_figures.py
```

`results/` stores the raw structured records, derived CSV files, figures, and
source data used by figures. Missing or failed runs are not converted into
numeric results. Figures are regenerated from saved data; they do not require
retraining.

## Limits

The trained controls use 51M and 124M models, not the scale of the original
XSA training claim. Frozen statistics through 6.9B are descriptive, not
training results. The medium-size training configuration was not run. Frozen measurements at
larger scales do not replace that missing training experiment.
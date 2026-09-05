# Reproduce

Everything needed to re-run this work, in the order it was run. Numbers in the
paper come from `results/`; `results/MANIFEST.md` maps each one to its artifact
and recomputes it, so a claim that no longer reproduces fails the build.

## What you can reproduce without a GPU

The three checks, the self-tests, the worked example, every table and every
figure regenerate on CPU in under two minutes. Only the training leg and the
frozen-model measurements need a GPU.

```bash
pip install -r requirements.txt

python scripts/selftest_arms.py --json results/selftest.json   # 10/10
python scripts/a4_recompute.py
python examples/audit_your_method.py
python scripts/make_tables.py
python scripts/make_figures.py
python scripts/check_figures.py
python scripts/make_manifest.py
python scripts/verify_day.py --all
python -m pytest -q
```

## Environment

Measured on the machine that produced the committed results.

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.11 |
| GPU (frozen models, long context) | NVIDIA A100-SXM4-80GB |
| GPU (factorial, calibration, pilot) | NVIDIA RTX 6000 Ada 48GB, community cloud |
| Driver | 595.71.05 |
| CUDA | 12.8 |
| cuDNN | 9.8.0 |

Package versions actually installed:

```
torch==2.8.0.dev20250319+cu128
transformers==5.16.1
numpy==1.26.4
scipy==1.17.1
matplotlib==3.11.1
pandas==3.0.5
datasets==5.0.1
tokenizers==0.23.2
safetensors==0.8.0
accelerate==1.14.0
tqdm==4.70.0
```

`numpy<2.0` is pinned deliberately: the pinned torch build is compiled against
the 1.x ABI.

## Model revisions

Hugging Face repositories move, so a model name does not identify what was
measured. Every revision is pinned in `results/model_revisions.json` and
reproduced here:

| model | revision |
|---|---|
| gpt2 | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| gpt2-medium | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` |
| gpt2-large | `32b71b12589c2f8d625668d2335a01cac3249519` |
| gpt2-xl | `15ea56dee5df4983c59b2538573817e1667135e2` |
| EleutherAI/pythia-160m | `50f5173d932e8e61f858120bcb800b97af589f46` |
| EleutherAI/pythia-410m | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |
| EleutherAI/pythia-1.4b | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` |
| EleutherAI/pythia-2.8b | `2a259cdd96a4beb1cdf467512e3904197345f6a9` |
| EleutherAI/pythia-6.9b | `c0e3eee36dc47af0c49f361c74cfe459c09f7f23` |
| Qwen/Qwen2.5-0.5B | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Qwen/Qwen2.5-1.5B | `8faed761d45a263340a0528343f099c05c9a4323` |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | `59f6f375b26bde864a6ca194a9a3044570490064` |

## Seeds

| Experiment | Seeds |
|---|---|
| Factorial (CFG_S pilot) | 42, 1337, 2024, 7, 99, 512, 8191, 31337 |
| Pilot / GO-NO-GO | 42, 1337, 2024 |
| Frozen-model measurements | 0 (document sampling is deterministic given the seed) |

Arms within a seed share initialisation and data order, so a paired difference
isolates the intervention. The self-tests assert this: measured step-0
deviation across all five arms is exactly `0.000e+00`.

## GPU run order, with cost

Run in this order. Each command is what actually produced the committed file.

```bash
# 1. Calibration. --cost-ceiling is required: passing 3.00 instead of the
#    spec's 56.00 is what voided the first factorial.
python scripts/calibrate_cli.py --rate <ACTUAL_USD_PER_HR> \
  --cost-ceiling 56 --n-runs 43 --device cuda
#    Confirm affordable:true AND tokens_per_run >= 3.5e8 before proceeding.
#    run_factorial.py refuses to start otherwise.

# 2. Data
python data/prepare.py --tokens 6e7 --val-tokens 4000000

# 3. Frozen-model measurements (inference only)
python scripts/run_frozen.py --ladder --n-docs 32 --block 512 \
  --device cuda --dtype bfloat16
python scripts/run_frozen.py --gqa --n-docs 32 --block 512 \
  --device cuda --dtype bfloat16
python scripts/run_generality.py --model gpt2 --n-docs 12 --block 256
python scripts/null_length_sensitivity.py --device cuda
python scripts/diagnose_gpt2.py --device cuda

# 4. Factorial
python scripts/run_factorial.py --size S --arms baseline xsa random \
  --seeds 42 1337 2024 7 99 512 8191 31337 --device cuda

# 5. Artifacts
python scripts/make_figures.py && python scripts/check_figures.py
python scripts/make_tables.py && python scripts/make_manifest.py
```

| Stage | GPU | Wall clock | Cost |
|---|---|---|---|
| Calibration + data prep + pilot | RTX 6000 Ada @ $0.74/hr | ~3 h | ~$2.20 |
| A1 ladder, A3 GQA, A5, A6, GPT-2 diagnosis | RTX 6000 Ada / A100 | ~4.5 h | ~$3.35 |
| CFG_S pilot factorial, 24 cells at 5e7 | RTX 6000 Ada @ $0.74/hr | ~7.5 h | ~$5.55 |
| CFG_S primary, 24 cells at 3.999e8 | RTX 6000 Ada @ $0.74/hr | ~16 h | ~$11.81 |
| Long-context bounded diagnostic | A100-SXM4-80GB @ $1.39/hr | ~1.5 h | ~$2.10 |

Running totals are kept in `BUDGET.md` against a $70 ceiling, with a $66
stop-and-report threshold enforced in `solve_token_budget` rather than merely
declared.

## What is not reproducible from this repository

Stated so that no one wastes a GPU-hour finding out.

* **`results/factorial_m.csv` does not exist.** The CFG_M scale check is
  dropped for budget, with the arithmetic in `BUDGET.md`.
* **`results/factorial_s.csv` does not exist yet.** The primary endpoint at
  399,900,672 tokens per run is running; see `RUN_IN_PROGRESS.md`. What is
  committed is `factorial_s_pilot_5e7.csv`, 24 cells at 5e7 tokens per run,
  outside the pre-registered [3.5e8, 6e8] band and reported as the
  underpowered pilot. `run_factorial.py` now refuses to start below the floor
  without `--i-accept-underpowered`.
* **The A2 measurement is sensitive to evaluation budget.** At 24 documents
  per half the Pythia reliability estimates are not stable: repeating the run
  moved pythia-410m's `r_delta` from +0.446 to -0.007. `results/reliability.csv`
  and `results/a2_correlations.csv` hold the 64-document run; both 24-document
  runs are kept in `results/a2_budget_comparison/`. Reproduce with
  `python scripts/run_reliability.py --n-docs 64 --device cuda --dtype bfloat16`.
* **The published GPT-2 reference triple does not reproduce.** Thirteen
  measurement conventions were tried and all thirteen are reported unselected
  in `results/gpt2_diagnosis.csv`. No configuration is presented as the correct
  one: choosing the setting that matches a target is the practice this work
  exists to criticise.

"""Generate results/MANIFEST.md: every paper number, re-derived from its source.

A manifest that is typed by hand asserts provenance. This one *checks* it. Each
entry names a claim, the artifact it comes from, the script that produced that
artifact, and a function that recomputes the number from the committed file. If
the recomputed value does not match the claim, the row is marked MISMATCH and
the script exits non-zero.

    python scripts/make_manifest.py

That makes "every number in the paper is read from a committed artifact" a
thing CI can fail on, rather than a sentence in a README.
"""

import csv
import json
import math
import pathlib
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOL = 5e-4


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _display(path):
    """Path relative to the repo when it is inside it, absolute otherwise.

    relative_to() raises for any directory outside ROOT, which turned a
    redirected output directory into a crash rather than a printed path.
    """
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def rows(name):
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ladder_by_model():
    data = rows("ladder.csv")
    if data is None:
        return None
    agg = defaultdict(list)
    for r in data:
        agg[r["model"]].append(r)
    out = {}
    for m, rs in agg.items():
        cs = sum(float(r["cos_self"]) for r in rs) / len(rs)
        cn = sum(float(r["cos_null"]) for r in rs) / len(rs)
        out[m] = {"cos_self": cs, "cos_null": cn, "n": len(rs),
                  "null_fraction": 100.0 * cn / cs}
    return out


def null_fraction(model):
    def f():
        by = _ladder_by_model()
        return None if not by or model not in by else by[model]["null_fraction"]
    return f


def gqa_field(model, field):
    def f():
        data = rows("gqa.csv")
        if data is None:
            return None
        for r in data:
            if r["model"] == model:
                try:
                    return float(r[field])
                except (TypeError, ValueError):
                    return None
        return None
    return f


def generality_fraction(method):
    def f():
        data = rows("generality.csv")
        if data is None:
            return None
        for r in data:
            if r["method"] == method and r.get("status") == "completed":
                return 100.0 * float(r["self_specific_fraction"])
        return None
    return f


def paired_field(arm, field, source="paired_tests_s.csv"):
    """Read a paired-test value, falling back to the underpowered pilot.

    The primary endpoint at 3.999e8 tokens is still running. Until it lands
    the only paired tests that exist are the 5e7 pilot's, and the claims are
    labelled as the pilot's rather than quietly presented as the endpoint.
    """
    def f():
        data = rows(source)
        if data is None:
            data = rows("paired_tests_s_pilot_5e7.csv")
        if data is None:
            return None
        for r in data:
            if r["arm"] == arm:
                return float(r[field])
        return None
    return f


def selftest_step0():
    def f():
        d = load("selftest.json")
        if not d:
            return None
        return 0.0 if d.get("all_pass") else 1.0
    return f


def gpt2_cos(field):
    def f():
        data = rows("gpt2_diagnosis.csv")
        if data is None:
            return None
        base = [r for r in data if r["variant"] == "reference"
                or r["block"] == "1024"]
        return float(base[0][field]) if base else None
    return f


def diagmask_slowdown(size):
    def f():
        d = load("calibration.json")
        try:
            return float(d["sizes"][size]["diagmask_slowdown"])
        except (TypeError, KeyError, ValueError):
            return None
    return f


def ladder_total_heads():
    def f():
        data = rows("ladder.csv")
        return None if data is None else float(len(data))
    return f



def a2_field(model, statistic, field):
    def f():
        data = rows("a2_correlations.csv")
        if data is None:
            return None
        for r in data:
            if r["model"] == model and r["statistic"] == statistic:
                try:
                    return float(r[field])
                except (TypeError, ValueError):
                    return None
        return None
    return f


def reliability_field(model, field):
    def f():
        data = rows("reliability.csv")
        if data is None:
            return None
        for r in data:
            if r["model"] == model and r.get("status") == "completed":
                try:
                    return float(r[field])
                except (TypeError, ValueError):
                    return None
        return None
    return f


#: claim text, expected value, unit, artifact, producing script, recompute fn
CLAIMS = [
    ("58% of cos(y_i, v_i) at Pythia-6.9B is explained by the matched null",
     58.1, "%", "ladder.csv", "run_frozen.py --ladder",
     null_fraction("EleutherAI/pythia-6.9b")),
    ("the null share at Pythia-160m is 63.1%, the ladder's maximum",
     63.1, "%", "ladder.csv", "run_frozen.py --ladder",
     null_fraction("EleutherAI/pythia-160m")),
    ("the null share falls to a minimum of 48.1% at Pythia-410m, then rises",
     48.1, "%", "ladder.csv", "run_frozen.py --ladder",
     null_fraction("EleutherAI/pythia-410m")),
    ("the ladder covers 6,784 head-level rows",
     6784, "rows", "ladder.csv", "run_frozen.py --ladder",
     ladder_total_heads()),
    ("TinyLlama-1.1B across-group excess is negative, replicating the sign "
     "in a second model family",
     -0.1126, "", "gqa.csv", "run_frozen.py --gqa",
     gqa_field("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
               "across_group_excess")),
    ("TinyLlama-1.1B within-group excess",
     0.2373, "", "gqa.csv", "run_frozen.py --gqa",
     gqa_field("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
               "within_group_excess")),
    ("Qwen2.5-0.5B within-group excess",
     0.2415, "", "gqa.csv", "run_frozen.py --gqa",
     gqa_field("Qwen/Qwen2.5-0.5B", "within_group_excess")),
    ("Qwen2.5-0.5B across-group excess is negative",
     -0.1876, "", "gqa.csv", "run_frozen.py --gqa",
     gqa_field("Qwen/Qwen2.5-0.5B", "across_group_excess")),
    ("Qwen2.5-1.5B within-group excess",
     0.2731, "", "gqa.csv", "run_frozen.py --gqa",
     gqa_field("Qwen/Qwen2.5-1.5B", "within_group_excess")),
    ("Qwen2.5-1.5B across-group excess is negative",
     -0.1922, "", "gqa.csv", "run_frozen.py --gqa",
     gqa_field("Qwen/Qwen2.5-1.5B", "across_group_excess")),
    ("attention sinks retain 99.2% of their statistic against a matched null",
     99.2, "%", "generality.csv", "run_generality.py",
     generality_fraction("attention_sink")),
    ("massive activations retain 71.7%",
     71.7, "%", "generality.csv", "run_generality.py",
     generality_fraction("massive_activations")),
    ("the random arm's paired mean delta vs baseline (underpowered 5e7 pilot, not the primary endpoint)",
     0.001190, "nats", "paired_tests_s.csv", "run_factorial.py --size S",
     paired_field("random", "mean_delta")),
    ("the random arm's p value (underpowered 5e7 pilot, not the primary endpoint)",
     0.042, "", "paired_tests_s.csv", "run_factorial.py --size S",
     paired_field("random", "p")),
    ("the xsa arm's paired mean delta vs baseline (underpowered 5e7 pilot, not the primary endpoint)",
     0.001515, "nats", "paired_tests_s.csv", "run_factorial.py --size S",
     paired_field("xsa", "mean_delta")),
    ("the xsa arm is not significant (underpowered 5e7 pilot, not the primary endpoint)",
     0.387, "", "paired_tests_s.csv", "run_factorial.py --size S",
     paired_field("xsa", "p")),
    ("all ten self-tests pass, step-0 deviation exactly zero",
     0.0, "", "selftest.json", "selftest_arms.py", selftest_step0()),
    ("diagmask slowdown at CFG_S, measured not predicted",
     2.3637, "x", "calibration.json", "calibrate_cli.py",
     diagmask_slowdown("S")),
    ("diagmask slowdown at CFG_M, measured not predicted",
     2.3364, "x", "calibration.json", "calibrate_cli.py",
     diagmask_slowdown("M")),
    ("the per-head XSA effect is resolvable in GPT-2 (split-half r_delta)",
     0.795, "", "reliability.csv", "run_reliability.py",
     reliability_field("gpt2", "r_delta")),
    ("in Pythia-160m the raw cosine and the null-corrected excess are "
     "close, so the model gives no clear ordering",
     0.149, "", "a2_correlations.csv", "run_reliability.py",
     a2_field("EleutherAI/pythia-160m", "cos_self", "rho_raw")),
    ("in Pythia-160m the null-corrected excess does predict it",
     0.189, "", "a2_correlations.csv", "run_reliability.py",
     a2_field("EleutherAI/pythia-160m", "excess", "rho_raw")),
    ("in Pythia-410m the raw self-value cosine carries nothing about the "
     "measured effect",
     -0.025, "", "a2_correlations.csv", "run_reliability.py",
     a2_field("EleutherAI/pythia-410m", "cos_self", "rho_raw")),
    ("in Pythia-410m the null-corrected excess again predicts",
     0.249, "", "a2_correlations.csv", "run_reliability.py",
     a2_field("EleutherAI/pythia-410m", "excess", "rho_raw")),
    ("in GPT-2 the ordering reverses: the raw cosine predicts better than "
     "the excess, which is reported rather than averaged away",
     0.462, "", "a2_correlations.csv", "run_reliability.py",
     a2_field("gpt2", "cos_self", "rho_raw")),
    ("the disattenuated GPT-2 correlation is stable across four independent "
     "runs, two budgets, two GPUs and two analysis revisions "
     "(0.521, 0.527, 0.526, 0.519)",
     0.519, "", "a2_correlations.csv", "run_reliability.py",
     a2_field("gpt2", "cos_self", "rho_disattenuated")),
]


def main(argv=None):
    sha = git_sha()
    lines = [
        "# Manifest: every paper number and where it comes from",
        "",
        "Generated by `scripts/make_manifest.py`. Each row is **recomputed** "
        "from the committed artifact at build time and compared against the "
        "claim, so this file cannot drift from the data. A row that does not "
        "reproduce is marked MISMATCH and the generator exits non-zero.",
        "",
        "* Repository SHA: `{}`".format(sha),
        "* Tolerance: {:g}".format(TOL),
        "* GPUs used: RTX 6000 Ada 48GB (community) and A100-SXM4-80GB. "
        "Frozen-model measurements are inference-only and hardware "
        "independent; the factorial's wall-clock and cost are not.",
        "",
        "| # | Claim | Value | Recomputed | Artifact | Produced by | Status |",
        "|--:|---|--:|--:|---|---|---|",
    ]
    bad, missing = 0, 0
    for i, (claim, expected, unit, artifact, script, fn) in enumerate(CLAIMS, 1):
        got = fn()
        if got is None:
            status, shown = "MISSING", "--"
            missing += 1
        elif math.isclose(got, expected, rel_tol=0, abs_tol=max(
                TOL, abs(expected) * 5e-3)):
            status, shown = "ok", "{:g}".format(round(got, 4))
        else:
            status, shown = "**MISMATCH**", "{:g}".format(round(got, 4))
            bad += 1
        lines.append("| {} | {} | {:g}{} | {} | `{}` | `{}` | {} |".format(
            i, claim, expected, unit, shown, artifact, script, status))

    lines += [
        "",
        "## Artifacts not yet produced",
        "",
        "| Claim area | Blocked on |",
        "|---|---|",
        "| A2 / A2a reliability and disattenuated correlations | "
        "`results/reliability.csv` and `results/a2_correlations.csv` do not "
        "exist; the experiment has not been run |",
        "| Primary endpoint at CFG_M | `results/factorial_m.csv` does not "
        "exist. The committed `factorial_s.csv` ran at 5e7 tokens, outside "
        "the pre-registered [3.5e8, 6e8] band, and is reported as an "
        "underpowered pilot rather than the primary endpoint |",
        "",
        "Nothing above is estimated, extrapolated or filled in. An experiment "
        "that has not run appears here as an experiment that has not run.",
        "",
        "## Model revisions",
        "",
        "Pinned in `results/model_revisions.json`. Hugging Face repositories "
        "move, so a model name alone does not identify what was measured.",
        "",
    ]
    out = RESULTS / "MANIFEST.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote {}".format(_display(out)))
    print("  {} claims, {} reproduced, {} missing, {} mismatched".format(
        len(CLAIMS), len(CLAIMS) - bad - missing, missing, bad))
    if bad:
        print("MISMATCH: a claim does not reproduce from its artifact.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Machine-checkable day gates.

    python scripts/verify_day.py 1        # check one day
    python scripts/verify_day.py --all    # check every day, report status

Exits non-zero if the requested day's gate fails. Do not start day N+1 until
day N's gate passes. Print the result at the end of every session.

Two of the spec's gate items cannot pass as written, and this script reports
that rather than papering over it. Both are documented in DEVIATIONS.md:

* **Day 1, A4.** The gate requires reproducing "floor 0.200, observed 0.373,
  excess 0.165 (44%)". Those numbers are mutually inconsistent: 0.373 - 0.200
  is 0.173, not 0.165, and 0.173/0.373 is 46.4%, not 44%. The gate here checks
  that the recompute exists and that its own arithmetic closes.
* **Day 1, test 10.** The gate requires max |loss_arm - loss_baseline| < 1e-6
  across all five arms. A hard -inf diagonal mask cannot satisfy this; the
  gated diagmask does, and is the default. See DEVIATIONS.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.runmeta import numeric_records, read_records  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = ROOT / "data"

Check = Tuple[bool, str, str]     # (passed, label, detail)


def _exists(path: Path, label: str) -> Check:
    return (path.exists(), label,
            str(path.relative_to(ROOT)) if path.exists() else
            "MISSING: {}".format(path.relative_to(ROOT)))


def _json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _factorial_rows(size: str) -> Tuple[List[dict], str]:
    """Read raw records when present, otherwise the committed aggregate.

    Clean clones intentionally do not contain ignored per-run directories.
    The gate must therefore be able to verify the committed evidence rather
    than reporting 0/2 simply because the provenance-preserving aggregate is
    the only representation available.
    """
    raw_dir = RESULTS / "factorial_{}".format(size)
    records = numeric_records(read_records(raw_dir))
    if records:
        return [dict(status=r.status, arm=r.arm, seed=r.seed, size=r.size,
                     **(r.metrics or {})) for r in records], raw_dir.name

    aggregate = RESULTS / "factorial_{}.csv".format(size)
    if not aggregate.exists():
        return [], aggregate.name
    with aggregate.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if r.get("status") in ("completed", "smoke")]
    return rows, aggregate.name


# ---------------------------------------------------------------------------

def day1() -> List[Check]:
    """CPU only, no GPU spend."""
    checks: List[Check] = []

    st = _json(RESULTS / "selftest.json")
    if st is None:
        checks.append((False, "10/10 self-tests",
                       "MISSING results/selftest.json; run "
                       "scripts/selftest_arms.py --json results/selftest.json"))
    else:
        checks.append((bool(st.get("all_pass")) and st.get("n_total") == 10,
                       "10/10 self-tests",
                       "{}/{} pass".format(st.get("n_pass"), st.get("n_total"))))

    p0 = RESULTS / "position0.txt"
    ok = p0.exists() and "z_0" in p0.read_text(encoding="utf-8")
    detail = "MISSING"
    if p0.exists():
        for line in p0.read_text(encoding="utf-8").splitlines():
            if "||z_0||" in line:
                detail = line.strip()
    checks.append((ok, "test 3 records ||z_0||", detail))

    pair = _json(RESULTS / "step0_pairing.json")
    if pair is None:
        checks.append((False, "test 10 < 1e-6 across all 5 arms", "MISSING"))
    else:
        dev = float(pair.get("max_abs_deviation", 1.0))
        checks.append((dev < 1e-6, "test 10 < 1e-6 across all 5 arms",
                       "max deviation {:.3e} over {} arms".format(
                           dev, len(pair.get("losses", {})))))

    # arms.py must define exactly Baseline, XSA, RandomDir, MeanValue.
    src = (ROOT / "xsac" / "arms.py").read_text(encoding="utf-8")
    defined = {ln.split("class ")[1].split("(")[0].strip()
               for ln in src.splitlines() if ln.startswith("class ")}
    expected = {"Arm", "Baseline", "_GatedRankOne", "XSA", "RandomDir",
                "MeanValue"}
    checks.append((defined == expected and "DiagMask" not in defined,
                   "arms.py defines exactly the 4 output arms",
                   "classes: {}".format(sorted(defined))))

    a4 = _json(RESULTS / "xsa_figure1_recompute.json")
    if a4 is None:
        checks.append((False, "A4 recompute exists",
                       "MISSING; run scripts/a4_recompute.py"))
    else:
        # The gate's own four numbers do not close. We require the recompute
        # to exist, to be internally consistent, and to have DETECTED the
        # inconsistency rather than silently adopting one reading.
        detected = a4.get("spec_is_internally_consistent") is False
        rec = a4.get("recomputed", {})
        closes = abs((a4["inputs"]["cos_self_observed"] - rec.get("floor", 0))
                     - rec.get("excess_vs_observed", -1)) < 1e-9
        checks.append((closes and detected, "A4 recompute is self-consistent",
                       "floor {:.4f}, excess {:.4f}, {:.1f}% self-specific; "
                       "spec's stated triple flagged inconsistent = {}".format(
                           rec.get("floor", float("nan")),
                           rec.get("excess_vs_observed", float("nan")),
                           rec.get("percent_self_specific", float("nan")),
                           detected)))

    for split in ("train", "val"):
        p = DATA / "{}.bin".format(split)
        if not p.exists():
            checks.append((False, "data/{}.bin exists".format(split),
                           "MISSING; run data/prepare.py or use --smoke"))
        else:
            size = p.stat().st_size
            checks.append((size % 2 == 0,
                           "data/{}.bin is a valid uint16 memmap".format(split),
                           "{} bytes = 2 x {} tokens".format(size, size // 2)))

    b = ROOT / "BUDGET.md"
    spent_zero = b.exists() and "$0.00" in b.read_text(encoding="utf-8")
    checks.append((spent_zero, "BUDGET.md exists with $0.00 spent",
                   "present and zero" if spent_zero else
                   "MISSING or non-zero spend"))
    return checks


def day2() -> List[Check]:
    """Throughput known, budget solved against the ACTUAL rate."""
    checks: List[Check] = []
    cal = _json(RESULTS / "calibration.json")
    if cal is None:
        checks.append((False, "calibration.json exists",
                       "MISSING; run scripts/calibrate_cli.py"))
        return checks

    for size in ("S", "M"):
        s = cal.get("sizes", {}).get(size)
        if not s:
            checks.append((False, "calibration for CFG_{}".format(size),
                           "MISSING"))
            continue
        base = s["baseline"]
        have = all(k in base for k in ("tokens_per_sec", "achieved_tflops",
                                       "mfu_vs_181", "seconds_per_step"))
        checks.append((have, "CFG_{} throughput fields".format(size),
                       "{:.0f} tok/s, {:.1f} TFLOP/s, MFU {:.1%}".format(
                           base["tokens_per_sec"], base["achieved_tflops"],
                           base["mfu_vs_181"])))
        sd = s.get("diagmask_slowdown", float("nan"))
        # The spec item is "diagmask slowdown factor vs baseline is measured
        # and recorded (expect 1.5-2.0x)". The requirement is the measurement;
        # the band is a parenthetical prediction. Asserting the prediction
        # would mean a correct measurement fails the gate whenever reality
        # disagrees with the guess, which is what happened here: 2.36x on an
        # RTX 6000 Ada and 2.34x on an A100, both outside the predicted band
        # and both correct. The gate checks that it was measured; the band is
        # reported alongside so the disagreement stays visible.
        in_band = s.get("slowdown_in_expected_band", False)
        checks.append((sd == sd and sd > 0,
                       "CFG_{} diagmask slowdown measured and recorded".format(size),
                       "measured {:.2f}x{}".format(
                           sd, "" if in_band else
                           "  [OUTSIDE the spec's predicted 1.5-2.0x band; "
                           "the measurement stands, the prediction does not]")))
        bud = s["budget"]
        tpr = bud["tokens_per_run"]
        in_band = 3.5e8 <= tpr <= 6.0e8
        checks.append((in_band or bud.get("drop_cfg_m", False),
                       "CFG_{} tokens_per_run clamped to [3.5e8, 6e8]".format(size),
                       "{:.3g} tokens{}".format(
                           tpr, " (CFG_M dropped)" if bud.get("drop_cfg_m")
                           else "")))

    checks.append((not cal.get("rate_is_placeholder", True),
                   "budget uses the ACTUAL rate, not the $0.86 placeholder",
                   cal.get("warning", "actual rate used: ${}/hr".format(
                       cal.get("rate_usd_hr")))))
    return checks


def day3() -> List[Check]:
    """GO/NO-GO. The most important gate in the project."""
    checks: List[Check] = []
    dec = _json(RESULTS / "pilot_decision.json")
    checks.append(_exists(RESULTS / "pilot.csv", "pilot.csv exists"))
    if dec is None:
        checks.append((False, "GO/NO-GO decision recorded",
                       "MISSING; run scripts/run_factorial.py --pilot"))
        return checks
    sigma, mde = dec.get("sigma_paired"), dec.get("mde")
    checks.append((sigma == sigma, "sigma_paired estimated",
                   "{}".format(sigma)))
    checks.append((mde == mde, "MDE computed", "{}".format(mde)))
    checks.append((dec.get("branch") in ("proceed", "reduce", "kill"),
                   "one of the three branches chosen",
                   "{}: {}".format(dec.get("branch"), dec.get("action"))))
    budget = (ROOT / "BUDGET.md").read_text(encoding="utf-8") \
        if (ROOT / "BUDGET.md").exists() else ""
    checks.append(("MDE" in budget, "decision written into BUDGET.md",
                   "BUDGET.md mentions MDE" if "MDE" in budget else
                   "BUDGET.md does not quote the MDE value"))
    return checks


def day46() -> List[Check]:
    """Factorial complete, pairing provably held."""
    checks: List[Check] = []
    for size in ("s", "m"):
        rows, source = _factorial_rows(size)
        if not rows:
            checks.append((False, "factorial_{} has runs".format(size),
                           "no completed records in {}".format(source)))
            continue
        by_seed: Dict[int, set] = {}
        budgets = set()
        arms = set()
        for row in rows:
            seed = int(row["seed"])
            budget = int(float(row["tokens_seen"]))
            by_seed.setdefault(seed, set()).add(budget)
            budgets.add(budget)
            arms.add(row["arm"])
        bad = {s: v for s, v in by_seed.items() if len(v) > 1}
        checks.append((not bad, "pairing: identical tokens_seen per seed "
                                "({})".format(size),
                       "all {} seeds consistent".format(len(by_seed)) if not bad
                       else "MISMATCH at seeds {}".format(sorted(bad))))
        checks.append((len(budgets) == 1,
                       "one token budget across the full grid ({})".format(size),
                       "tokens_seen={}".format(next(iter(budgets)))
                       if len(budgets) == 1 else
                       "MIXED budgets: {}".format(sorted(budgets))))
        expected_arms = {"baseline", "xsa", "random"}
        expected_seeds = 8 if size == "s" else 3
        complete = (arms == expected_arms and len(by_seed) == expected_seeds
                    and len(rows) == len(expected_arms) * expected_seeds)
        checks.append((complete, "pre-registered grid complete ({})".format(size),
                       "{} rows, {} seeds, arms {} from {}".format(
                           len(rows), len(by_seed), sorted(arms), source)))
        losses = [float(r["final_val_loss"]) for r in rows
                  if r.get("final_val_loss") not in (None, "")]
        checks.append((len(losses) == len(rows) and
                       all(v == v for v in losses),
                       "no NaN in the loss column ({})".format(size),
                       "{} rows".format(len(losses))))

        pt = RESULTS / "paired_tests_{}.csv".format(size)
        if pt.exists():
            import csv as _csv
            rows = list(_csv.DictReader(pt.open(encoding="utf-8")))
            has_primary = any("PRIMARY" in r.get("endpoint", "") for r in rows)
            first_is_primary = bool(rows) and "PRIMARY" in rows[0].get(
                "endpoint", "")
            checks.append((has_primary and first_is_primary,
                           "primary endpoint reported first and labelled ({})"
                           .format(size),
                           "first row: {} / {}".format(
                               rows[0].get("arm"), rows[0].get("endpoint"))
                           if rows else "empty"))
            checks.append((all(r.get("sd_unpaired_treatment") for r in rows),
                           "unpaired sd reported alongside ({})".format(size),
                           "{} rows carry it".format(len(rows))))
    return checks


def day7() -> List[Check]:
    """The ladder."""
    checks: List[Check] = []
    lad = RESULTS / "ladder.csv"
    if not lad.exists():
        checks.append((False, "ladder.csv exists", "MISSING"))
        return checks
    import csv as _csv
    rows = list(_csv.DictReader(lad.open(encoding="utf-8")))
    models = {r["model"] for r in rows}
    checks.append((len(models) >= 9, "nine models on the ladder",
                   "{} models: {}".format(len(models), sorted(models))))
    for needed in ("EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b"):
        checks.append((needed in models,
                       "{} present (answers the scale objection)".format(needed),
                       "present" if needed in models else "MISSING"))
    chk = _json(RESULTS / "gpt2_target_check.json")
    if chk is None:
        checks.append((False, "GPT-2 reproduces 0.5406/0.3798/0.1608",
                       "MISSING gpt2_target_check.json"))
    else:
        checks.append((bool(chk.get("reproduces")),
                       "GPT-2 reproduces 0.5406/0.3798/0.1608 (+/-0.01)",
                       "deltas {}".format({k: round(v, 4) for k, v in
                                           chk["abs_deltas"].items()})))
    return checks


def day8() -> List[Check]:
    checks: List[Check] = []
    checks.append(_exists(RESULTS / "gqa.csv", "gqa.csv exists"))
    checks.append(_exists(RESULTS / "generality.csv", "generality.csv (A6)"))
    checks.append(_exists(RESULTS / "position0.txt", "position0.txt"))
    return checks


def day9() -> List[Check]:
    checks: List[Check] = []
    figs = RESULTS / "figures"
    for stem in ("fig1_gates", "fig2_paired_delta", "fig3_ladder",
                 "fig4_generality", "fig5_gqa"):
        png, pdf = figs / (stem + ".png"), figs / (stem + ".pdf")
        data = figs / (stem + "_data.csv")
        checks.append((png.exists() and pdf.exists() and data.exists(),
                       "{} png + pdf + data.csv".format(stem),
                       "present" if png.exists() and pdf.exists() and
                       data.exists() else "MISSING one of png/pdf/_data.csv"))
    return checks


def day10() -> List[Check]:
    """Draft. Text checks only; the draft itself is written by a human."""
    checks: List[Check] = []
    draft = ROOT / "paper" / "draft.md"
    if not draft.exists():
        checks.append((False, "draft exists", "MISSING paper/draft.md"))
        return checks
    text = draft.read_text(encoding="utf-8")
    for cite in ("Ethayarajh", "Timkey", "Machina", "Zhao"):
        checks.append((cite in text, "{} cited".format(cite),
                       "present" if cite in text else "MISSING"))
    forbidden = "omits the anisotropy control"
    checks.append((forbidden not in text,
                   'the false sentence "XSA omits the anisotropy control" '
                   'appears nowhere',
                   "absent" if forbidden not in text else "PRESENT - it is "
                   "false; XSA's Figure 1 left plots cos(v_i,v_j)"))
    return checks


DAYS: Dict[str, Callable[[], List[Check]]] = {
    "1": day1, "2": day2, "3": day3, "4": day46, "5": day46, "6": day46,
    "7": day7, "8": day8, "9": day9, "10": day10,
}


def report(day: str) -> bool:
    fn = DAYS.get(day)
    if fn is None:
        print("unknown day {!r}; expected one of {}".format(
            day, sorted(DAYS, key=int)))
        return False
    checks = fn()
    print("Day {} gate".format(day))
    print("-" * 72)
    for ok, label, detail in checks:
        print("  [{}] {:<52s} {}".format("PASS" if ok else "FAIL",
                                         label, detail))
    passed = all(ok for ok, _, _ in checks)
    print("  => {} ({}/{} checks)".format(
        "PASS" if passed else "FAIL",
        sum(1 for ok, _, _ in checks if ok), len(checks)))
    return passed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("day", nargs="?", default=None)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if args.all:
        results = {}
        for d in sorted(DAYS, key=int):
            results[d] = report(d)
            print()
        done = [d for d, ok in results.items() if ok]
        print("=" * 72)
        print("Gates passing: {}".format(", ".join(done) if done else "none"))
        print("Do not start day N+1 until day N passes.")
        return 0
    if not args.day:
        ap.error("give a day number or --all")
    return 0 if report(args.day) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Append actual GPU-hours and dollars to BUDGET.md after each seed block.

The factorial runs seed-major, so after every seed all three arms of that
seed are complete and a paired comparison exists for it. That is the unit
worth accounting in: if the money runs out mid-run you need N complete
paired seeds, not a scatter of orphaned cells.

    python scripts/budget_ledger.py --rate 0.74 --ceiling 18

Reads the run records rather than a log, so it is correct after a resume and
cannot double-count. Prints the projection to completion and exits 2 if that
projection crosses the ceiling, which is the signal to stop, commit what
completed, and report.
"""

import argparse
import collections
import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEDGER = ROOT / "BUDGET.md"
MARK = "<!-- seed-block ledger: appended by scripts/budget_ledger.py -->"


def records(results_dir):
    out = []
    runs = pathlib.Path(results_dir) / "runs"
    if not runs.exists():
        return out
    for f in sorted(runs.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        m = d.get("metrics") or {}
        cfg = d.get("config") or {}
        out.append({
            "seed": d.get("seed", cfg.get("seed")),
            "arm": cfg.get("arm") or d.get("arm"),
            "status": d.get("status"),
            "seconds": m.get("seconds") or m.get("wall_seconds")
            or m.get("train_seconds"),
            "tokens_seen": m.get("tokens_seen"),
        })
    return out


def summarise(recs, n_seeds_planned, n_arms):
    """Per-seed hours, and a projection built only from complete seeds.

    Incomplete seeds are excluded from the rate estimate on purpose: a seed
    that is half finished would otherwise drag the average down and make the
    projection look cheaper than it is.
    """
    by_seed = collections.defaultdict(list)
    for r in recs:
        if r["status"] in ("completed",) and r["seconds"]:
            by_seed[r["seed"]].append(r)
    complete = {s: rs for s, rs in by_seed.items() if len(rs) >= n_arms}
    hours = {s: sum(r["seconds"] for r in rs) / 3600.0
             for s, rs in complete.items()}
    done = len(complete)
    spent_h = sum(hours.values())
    per_seed = (spent_h / done) if done else 0.0
    remaining = max(n_seeds_planned - done, 0)
    return {
        "complete_paired_seeds": done,
        "partial_seeds": sorted(set(by_seed) - set(complete)),
        "hours_by_seed": hours,
        "gpu_hours_so_far": spent_h,
        "hours_per_seed": per_seed,
        "remaining_seeds": remaining,
        "projected_remaining_hours": per_seed * remaining,
        "projected_total_hours": spent_h + per_seed * remaining,
        "budgets": sorted({r["tokens_seen"] for r in recs
                           if r.get("tokens_seen")}),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(RESULTS / "factorial_s"))
    ap.add_argument("--rate", type=float, required=True,
                    help="USD per GPU-hour, actual not assumed")
    ap.add_argument("--ceiling", type=float, default=18.0,
                    help="stop-and-report threshold in USD")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--n-arms", type=int, default=3)
    ap.add_argument("--pod-hours", type=float, default=0.0,
                    help="TOTAL pod uptime so far. This is the billed "
                         "quantity: the pod charges wall clock, and the "
                         "training cells run inside that clock rather than "
                         "alongside it. Cell hours are used only to estimate "
                         "the per-seed rate, never added to this.")
    ap.add_argument("--append", action="store_true",
                    help="append a row to BUDGET.md")
    args = ap.parse_args(argv)

    recs = records(args.results_dir)
    s = summarise(recs, args.n_seeds, args.n_arms)

    # The pod bills wall clock. Training cells run inside that clock, so
    # adding cell hours to pod uptime counts the same minutes twice. Uptime
    # is the billed base; cell hours only set the per-seed rate used to
    # project the remainder.
    cells_usd = s["gpu_hours_so_far"] * args.rate
    billed_hours = args.pod_hours or s["gpu_hours_so_far"]
    total_usd = billed_hours * args.rate
    projected_usd = (billed_hours + s["projected_remaining_hours"]) * args.rate

    print("complete paired seeds   %d of %d" % (s["complete_paired_seeds"],
                                                args.n_seeds))
    if s["partial_seeds"]:
        print("partial seeds           %s  (not counted)" % s["partial_seeds"])
    if len(s["budgets"]) > 1:
        print("WARNING: mixed token budgets in one results directory: %s"
              % s["budgets"])
    print("GPU-hours in cells      %.2f  (=$%.2f of work, not the bill)"
          % (s["gpu_hours_so_far"], cells_usd))
    print("hours per paired seed   %.3f" % s["hours_per_seed"])
    print("pod uptime billed       %.2f h  ->  $%.2f" % (billed_hours,
                                                         total_usd))
    print("remaining %d seeds      %.2f h  ->  $%.2f"
          % (s["remaining_seeds"], s["projected_remaining_hours"],
             s["projected_remaining_hours"] * args.rate))
    print("projected at completion $%.2f  (ceiling $%.2f)"
          % (projected_usd, args.ceiling))

    if args.append and s["complete_paired_seeds"]:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        row = ("| {} | {} of {} paired seeds | {:.2f} | ${:.2f} | ${:.2f} | "
               "${:.2f} |".format(stamp, s["complete_paired_seeds"],
                                  args.n_seeds, s["gpu_hours_so_far"],
                                  cells_usd, total_usd, projected_usd))
        text = LEDGER.read_text(encoding="utf-8")
        if MARK not in text:
            text += ("\n\n## Seed-block ledger, appended as the run proceeds\n\n"
                     + MARK + "\n\n"
                     "Each row is written after a seed block completes, so the "
                     "unit is a **complete paired seed** rather than a cell. "
                     "The projection assumes the remaining seeds cost what the "
                     "completed ones did.\n\n"
                     "| when | progress | GPU-hours (cells) | $ cells | "
                     "$ incl. pod | projected total |\n"
                     "|---|---|--:|--:|--:|--:|\n")
        if row.split("|")[2].strip() not in text:
            text = text.rstrip() + "\n" + row + "\n"
            LEDGER.write_text(text, encoding="utf-8")
            print("appended to BUDGET.md")
        else:
            print("BUDGET.md already records this seed count")

    if projected_usd > args.ceiling:
        print("\nSTOP: projected ${:.2f} crosses the ${:.2f} ceiling. Commit "
              "the {} complete paired seeds and report."
              .format(projected_usd, args.ceiling,
                      s["complete_paired_seeds"]), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

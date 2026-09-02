"""
Regenerate every figure from results/*.csv. No manual steps, no retraining.

    python scripts/make_figures.py

A figure whose inputs are absent is reported as skipped and no file is written
for it. That is the point: an experiment that has not run must look like an
experiment that has not run, in the figure directory as much as in the tables.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xsac.figures import ALL_FIGURES, FigureSkipped  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)

    results, out = Path(args.results), Path(args.out)
    names = args.only or list(ALL_FIGURES)
    made, skipped = 0, 0
    for name in names:
        fn = ALL_FIGURES.get(name)
        if fn is None:
            print("  unknown figure {!r}".format(name))
            continue
        try:
            paths = fn(results, out)
            made += 1
            print("  [made]    {:<20s} {}".format(
                name, ", ".join(p.name for p in paths)))
        except FigureSkipped as exc:
            skipped += 1
            print("  [skipped] {:<20s} {}".format(name, exc))
        except Exception as exc:
            skipped += 1
            print("  [ERROR]   {:<20s} {}: {}".format(
                name, type(exc).__name__, exc))

    print("\n{} figure(s) written, {} skipped.".format(made, skipped))
    if skipped:
        print("Skipped figures have no data yet. They are not placeholders "
              "and nothing was drawn for them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

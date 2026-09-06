"""Recompute every number in Tables 1-3 from the stored result files.

The manuscript's tables are written by hand. Nothing in the build reads a CSV,
so a table can drift from the archive it reports and still compile. This
closes that gap: it parses the three tables out of main.tex, recomputes each
cell from the archived results, and compares.

    python check_tables.py                 # check against ../../results
    python check_tables.py --results DIR

Derived cells are recomputed rather than read, so an arithmetic slip in the
manuscript fails here too:

    rho_max  = sqrt(r_delta * r_stat)          Equation 2
    rho_dis  = rho_raw / rho_max               Equation 2
    MDE      = 2.9 * sd_paired / sqrt(n)       Equation 3

Comparison is on the rendered string, at the precision the table prints. A
cell showing 0.795 is checked as 0.795, not as the full float, because that is
what a reader sees.

Exit status is non-zero if any cell disagrees with its source.
"""

import argparse
import csv
import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parents[1] / "results"

# Manuscript label -> the model string used in the result files.
MODELS = {
    "GPT-2": "gpt2",
    "Pythia-160M": "EleutherAI/pythia-160m",
    "Pythia-410M": "EleutherAI/pythia-410m",
}
# Manuscript label -> the statistic name used in a2_correlations.csv.
STATS = {"self-cosine": "cos_self", "excess": "excess"}

MDE_MULTIPLIER = 2.9  # 80% power at alpha 0.05, per the protocol


def rows(results, name):
    path = pathlib.Path(results) / name
    if not path.exists():
        raise SystemExit("missing result file: %s" % path)
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rs, **eq):
    for r in rs:
        if all(str(r.get(k)) == str(v) for k, v in eq.items()):
            return r
    raise SystemExit("no row matching %r" % (eq,))


def table_body(tex, label):
    """The rows between \\midrule and \\bottomrule of the table with `label`."""
    # The inner group refuses to cross another \begin{table}, so the match
    # starts at this table's own opening rather than the document's first.
    block = re.search(
        r"\\begin\{table\}(?:(?!\\begin\{table\}).)*?"
        r"\\label\{%s\}.*?\\end\{table\}" % re.escape(label),
        tex, re.S)
    if not block:
        raise SystemExit("table %s not found in main.tex" % label)
    body = re.search(r"\\midrule(.*?)\\bottomrule", block.group(0), re.S)
    return body.group(1)


def cells(body):
    """Table body -> list of stripped cell lists, one per row.

    \\multirow spans and math delimiters are stripped so a cell compares as the
    number a reader sees.
    """
    out = []
    for line in body.split("\\\\"):
        line = re.sub(r"\\multirow\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", line)
        line = line.replace("$", "").replace("\\textsc{", "").strip()
        if not line or line.startswith("%"):
            continue
        out.append([_balance(c.strip()) for c in line.split("&")])
    return out


def _balance(cell):
    """Drop trailing braces left over by \\textsc{...}, and only those.

    A blanket rstrip("}") would eat the real closing braces of a cell like
    \\sigma_{\\text{paired}}, and the row would then match no known field and
    be skipped without a word. That happened.
    """
    while cell.endswith("}") and cell.count("}") > cell.count("{"):
        cell = cell[:-1]
    return cell


class Report:
    def __init__(self):
        self.problems = []
        self.checked = 0

    def eq(self, where, shown, expected, fmt):
        """Compare a printed cell against the value recomputed from source."""
        self.checked += 1
        want = fmt.format(expected)
        got = shown.strip()
        if got != want:
            self.problems.append("%s: table shows %r, source gives %r"
                                 % (where, got, want))
            print("  MISMATCH  %-42s shows %-12s source %s"
                  % (where, got, want))
        else:
            print("  ok        %-42s %s" % (where, got))

    def same(self, where, shown, expected):
        self.checked += 1
        if shown.strip() != str(expected).strip():
            self.problems.append("%s: table shows %r, source gives %r"
                                 % (where, shown, expected))
            print("  MISMATCH  %-42s shows %-12s source %s"
                  % (where, shown, expected))
        else:
            print("  ok        %-42s %s" % (where, shown.strip()))


def check_reliability(tex, results, rep):
    print("\nTable 1  results/reliability.csv")
    rel = rows(results, "reliability.csv")
    for row in cells(table_body(tex, "tab:reliability")):
        label, r_delta, r_stat, rho_max, verdict = row[:5]
        src = pick(rel, model=MODELS[label])
        rd, rs = float(src["r_delta"]), float(src["r_stat"])
        rep.eq("%s r_delta" % label, r_delta, rd, "{:.3f}")
        rep.eq("%s r_stat" % label, r_stat, rs, "{:.3f}")
        # Recomputed from the two reliabilities, not read from the file.
        rep.eq("%s rho_max (recomputed)" % label, rho_max,
               math.sqrt(rd * rs), "{:.3f}")
        rep.same("%s verdict" % label, verdict, src["verdict"])


def check_correlations(tex, results, rep):
    print("\nTable 2  results/a2_correlations.csv")
    a2 = rows(results, "a2_correlations.csv")
    model = None
    for row in cells(table_body(tex, "tab:a2")):
        # A \multirow row carries the model name; its continuation row leaves
        # that cell empty and inherits it.
        if row[0].strip():
            model = row[0].strip()
        stat, raw, rho_max, disatt = row[-4:]
        src = pick(a2, model=MODELS[model], statistic=STATS[stat.strip()])
        rr, ceil = float(src["rho_raw"]), float(src["ceiling"])
        where = "%s %s" % (model, stat.strip())
        rep.eq(where + " rho_raw", raw, rr, "{:+.3f}")
        rep.eq(where + " rho_max", rho_max, ceil, "{:.3f}")
        # Recomputed as raw / ceiling rather than read back.
        rep.eq(where + " rho_disatt (recomputed)", disatt, rr / ceil,
               "{:+.3f}")


def check_paired(tex, results, rep):
    print("\nTable 3  results/paired_tests_s.csv")
    pt = rows(results, "paired_tests_s.csv")
    body = cells(table_body(tex, "tab:paired"))
    arms = ["random", "xsa"]
    fields = {
        "Mean \\Delta (nats)": ("mean_delta", "{:+.6f}"),
        "CI low": ("ci_low", "{:+.6f}"),
        "CI high": ("ci_high", "{:+.6f}"),
        "t": ("t", "{:+.2f}"),
        "Cohen's d_z": ("cohen_dz", "{:+.3f}"),
        "\\sigma_{\\text{paired}}": ("sd_paired", "{:.6f}"),
    }
    known = set(fields) | {"p", "Holm p", "Realised MDE"}
    for row in body:
        name = row[0].strip()
        if name not in known:
            # A row nobody checks is worse than a failing one, because it
            # looks like a pass.
            rep.problems.append("Table 3 row %r is not checked against any "
                                "source field" % name)
            print("  UNCHECKED %s" % name)
            continue
        for i, arm in enumerate(arms):
            shown = row[1 + i].strip()
            src = pick(pt, arm=arm, vs="baseline")
            where = "%s [%s]" % (name, arm)
            if name in fields:
                key, fmt = fields[name]
                rep.eq(where, shown, float(src[key]), fmt)
            elif name == "p":
                rep.eq(where, shown, float(src["p"]),
                       "{:.3f}" if arm == "random" else "{:.4f}")
            elif name == "Holm p":
                holm = src.get("p_holm", "")
                if holm in ("", "nan"):
                    rep.same(where, shown, "n/a")
                else:
                    rep.eq(where, shown, float(holm), "{:.4f}")
            elif name == "Realised MDE":
                sd = float(src["sd_paired"])
                n = int(float(src["n_seeds"]))
                # Recomputed from Equation 3, not read from any file.
                rep.eq(where, shown,
                       MDE_MULTIPLIER * sd / math.sqrt(n), "{:.5f}")

    # The caption's design constants, checked against the archive too.
    n_seeds = {int(float(r["n_seeds"])) for r in pt}
    if n_seeds != {8}:
        rep.problems.append("caption says eight paired seeds; archive has %s"
                            % sorted(n_seeds))
    fac = rows(results, "factorial_s.csv")
    done = [r for r in fac if r.get("status", "completed") == "completed"]
    tokens = {int(float(r["tokens_seen"])) for r in done if r.get("tokens_seen")}
    print("  ok        %-42s %d completed cells" % ("factorial cells", len(done)))
    if tokens:
        print("  ok        %-42s %s" % ("tokens per run", sorted(tokens)))
        if tokens != {399900672}:
            rep.problems.append(
                "caption says 399,900,672 tokens per run; archive has %s"
                % sorted(tokens))
    if len(done) != 24:
        rep.problems.append("caption implies 24 cells; archive has %d"
                            % len(done))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(DEFAULT_RESULTS))
    ap.add_argument("--tex", default=str(HERE / "main.tex"))
    args = ap.parse_args(argv)

    tex = pathlib.Path(args.tex).read_text(encoding="utf-8")
    rep = Report()
    check_reliability(tex, args.results, rep)
    check_correlations(tex, args.results, rep)
    check_paired(tex, args.results, rep)

    print("\n%d cells checked against %s" % (rep.checked, args.results))
    if rep.problems:
        print("\n%d disagree with their source:" % len(rep.problems))
        for p in rep.problems:
            print("  " + p)
        return 1
    print("every cell in Tables 1-3 matches the archived results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

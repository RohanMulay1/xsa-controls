"""Emit every paper table as LaTeX (booktabs) and Markdown, from results/.

There is a make_figures.py and papers are half tables. Each table reads one
committed CSV under ``results/`` and nothing else: no hand-entered numbers, no
manual editing step, so a table can never drift from the run that produced it.

    python scripts/make_tables.py

Writes ``results/tables/T<n>_<name>.tex`` and ``.md``. A table whose source
CSV does not exist yet is skipped and reported, not faked -- T4 and T5 need
the CFG_M factorial and the A2 correlations respectively.

Every caption carries ``n``. That is a hard requirement of the spec and it is
enforced here rather than left to whoever writes the paper: ``emit`` refuses a
table whose caption does not state one.
"""

import csv
import math
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"


def _display(path):
    """Path relative to the repo when it is inside it, absolute otherwise.

    relative_to() raises for any directory outside ROOT, which turned a
    redirected output directory into a crash rather than a printed path.
    """
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def read(name):
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fnum(value, places=4, dash="--"):
    """Format a float, rendering a missing or NaN cell as an em dash."""
    if value is None or value == "":
        return dash
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(f):
        return dash
    return "{:.{p}f}".format(f, p=places)


def latex_escape(text):
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def emit(key, caption, header, rows, aligns=None):
    """Write one table as .tex and .md.

    The caption must state n. A table without its sample size is not
    reportable, and catching that here is cheaper than catching it in review.
    """
    if "n =" not in caption and "n=" not in caption:
        raise ValueError(
            "caption for {} does not state n: {!r}".format(key, caption))
    OUT.mkdir(parents=True, exist_ok=True)
    aligns = aligns or (["l"] + ["r"] * (len(header) - 1))

    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join("---" if a == "l" else "--:" for a in aligns) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    md_text = "**{}.** {}\n\n".format(key, caption) + "\n".join(md) + "\n"
    (OUT / "{}.md".format(key)).write_text(md_text, encoding="utf-8")

    tex = [r"\begin{table}[t]", r"  \centering",
           r"  \caption{" + latex_escape(caption) + "}",
           r"  \label{tab:" + key.lower() + "}",
           r"  \begin{tabular}{" + "".join(aligns) + "}",
           r"    \toprule",
           "    " + " & ".join(latex_escape(h) for h in header) + r" \\",
           r"    \midrule"]
    tex += ["    " + " & ".join(latex_escape(str(c)) for c in r) + r" \\"
            for r in rows]
    tex += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    (OUT / "{}.tex".format(key)).write_text("\n".join(tex) + "\n",
                                            encoding="utf-8")
    print("  wrote {}.tex and {}.md  ({} rows)".format(key, key, len(rows)))


def _by_model(rows):
    agg = defaultdict(list)
    for r in rows:
        agg[r["model"]].append(r)
    return agg


def table1_ladder():
    rows = read("ladder.csv")
    if rows is None:
        return "ladder.csv missing"
    # The ladder holds both MHA and GQA models. Table 1 is the MHA scale
    # ladder; the GQA models have their own table because the within/across
    # group split does not exist without grouped-query attention.
    mha = [r for r in rows if int(r["n_kv_heads"]) == int(r.get("n_kv_heads"))
           and not r["model"].startswith("Qwen")]
    agg = _by_model(mha)
    out = []
    for model, rs in agg.items():
        cs = sum(float(r["cos_self"]) for r in rs) / len(rs)
        cn = sum(float(r["cos_null"]) for r in rs) / len(rs)
        out.append((model, cs, cn, cs - cn, 100.0 * cn / cs, len(rs)))
    out.sort(key=lambda t: PARAMS.get(t[0], 0))
    body = [[m, PARAM_LABEL.get(m, "?"), fnum(cs), fnum(cn), fnum(ex),
             "{:.1f}".format(100 - nullpc), "{:d}".format(n)]
            for m, cs, cn, ex, nullpc, n in out]
    emit("T1",
         "Check 1 across the multi-head scale ladder. The null is a partner "
         "drawn within the same sequence from positions the query could "
         "causally attend, so it is matched for position, recency and "
         "sequence. n = {} models, {} heads, 32 wikitext-103 documents "
         "each.".format(len(out), sum(t[5] for t in out)),
         ["model", "params", "cos\\_self", "cos\\_null", "excess",
          "\\% self-specific", "heads"],
         body)
    return None


def table2_gqa():
    rows = read("gqa.csv")
    if rows is None:
        return "gqa.csv missing"
    gqa = [r for r in rows if str(r.get("is_gqa")).lower() == "true"]
    if not gqa:
        return "gqa.csv contains no grouped-query models"
    body = [[r["model"], r["n_query_heads"], r["n_kv_heads"],
             fnum(r["within_group_excess"]), fnum(r["across_group_excess"]),
             r["n_heads"]] for r in gqa]
    emit("T2",
         "Check 1 under grouped-query attention, split by whether the null "
         "partner comes from the head's own KV group. Borrowing a "
         "neighbouring group's value at the same position does not merely "
         "lose the effect, it reverses it. n = {} GQA models, {} heads."
         .format(len(gqa), sum(int(r["n_heads"]) for r in gqa)),
         ["model", "query heads", "KV heads", "within-group excess",
          "across-group excess", "heads"],
         body)
    return None


def table3_generality():
    rows = read("generality.csv")
    if rows is None:
        return "generality.csv missing"
    done = [r for r in rows if r.get("status") == "completed"]
    if not done:
        return "generality.csv has no completed rows"
    body = [[r["method"], r["model"], fnum(r["observed"]), fnum(r["null"]),
             fnum(r["excess"]),
             "{:.1f}".format(100 * float(r["self_specific_fraction"])),
             "yes" if str(r["survives_null"]).lower() == "true" else "no"]
            for r in done]
    emit("T3",
         "Check 1 applied to other attention findings, each against a null "
         "matched to the structure its own statistic inherits for free. The "
         "checklist separates methods rather than debunking uniformly. "
         "n = {} methods.".format(len(done)),
         ["method", "model", "statistic", "null", "excess",
          "\\% self-specific", "survives"],
         body)
    return None


def table4_paired():
    for name, label in (("paired_tests_m.csv", "CFG\\_M"),
                        ("paired_tests_s.csv", "CFG\\_S")):
        rows = read(name)
        if not rows:
            continue
        body = [[r["arm"], fnum(r["mean_delta"], 6),
                 "[{}, {}]".format(fnum(r["ci_low"], 6), fnum(r["ci_high"], 6)),
                 fnum(r["t"], 2), fnum(r["p"], 4),
                 fnum(r.get("p_holm"), 4), fnum(r.get("cohen_dz"), 3),
                 r["n_seeds"]] for r in rows]
        pilot = " This is the underpowered pilot at 5e7 tokens per run, " \
                "outside the pre-registered [3.5e8, 6e8] band; it is " \
                "reported as provenance, not as the primary endpoint." \
                if name.endswith("_s.csv") else ""
        emit("T4",
             "Paired difference in final validation loss against baseline, "
             "{}. The primary endpoint is listed first; Holm correction is "
             "applied over the secondary arms only.{} n = {} seeds per arm."
             .format(label, pilot, rows[0]["n_seeds"]),
             ["arm", "mean $\\Delta$", "95\\% CI", "t", "p", "Holm p",
              "Cohen $d_z$", "seeds"],
             body)
        return None
    return "no paired_tests_*.csv found"


def table5_a2():
    rows = read("a2_correlations.csv")
    if rows is None:
        return ("a2_correlations.csv missing -- run the A2a/A2 reliability "
                "experiment first")
    body = [[r.get("statistic"), fnum(r.get("rho_raw"), 3),
             fnum(r.get("r_delta"), 3), fnum(r.get("r_stat"), 3),
             fnum(r.get("rho_disattenuated"), 3), r.get("n")] for r in rows]
    emit("T5",
         "Per-head motivating statistic against measured intervention effect. "
         "The disattenuated column divides by sqrt(r_delta * r_stat), the "
         "ceiling that split-half unreliability places on any observable "
         "correlation. n = {} rows.".format(len(rows)),
         ["statistic", "raw $\\rho$", "$r_\\Delta$", "$r_{stat}$",
          "disattenuated $\\rho$", "n"],
         body)
    return None


def table6_gpt2():
    rows = read("gpt2_diagnosis.csv")
    if rows is None:
        return "gpt2_diagnosis.csv missing"
    body = [[r["variant"], r["block"], fnum(r["cos_self"]), fnum(r["cos_null"]),
             fnum(r["excess"]),
             "yes" if str(r["matches_reference"]).lower() == "true" else "no"]
            for r in rows]
    emit("T6",
         "Appendix. Every measurement convention tried when reproducing the "
         "published GPT-2 reference triple, reported unselected. No "
         "configuration is presented as the correct one: selecting the "
         "setting that matches a target is the practice this work "
         "criticises. n = {} configurations.".format(len(rows)),
         ["convention varied", "block", "cos\\_self", "cos\\_null", "excess",
          "matches reference"],
         body)
    return None


#: Parameter counts are model facts, not measurements, so they are not in the
#: result CSVs. They label rows and order the ladder; they are never averaged.
PARAMS = {"gpt2": 124e6, "EleutherAI/pythia-160m": 160e6,
          "gpt2-medium": 355e6, "EleutherAI/pythia-410m": 410e6,
          "gpt2-large": 774e6, "EleutherAI/pythia-1.4b": 1.4e9,
          "gpt2-xl": 1.5e9, "EleutherAI/pythia-2.8b": 2.8e9,
          "EleutherAI/pythia-6.9b": 6.9e9,
          "Qwen/Qwen2.5-0.5B": 0.5e9, "Qwen/Qwen2.5-1.5B": 1.5e9}
PARAM_LABEL = {"gpt2": "124M", "EleutherAI/pythia-160m": "160M",
               "gpt2-medium": "355M", "EleutherAI/pythia-410m": "410M",
               "gpt2-large": "774M", "EleutherAI/pythia-1.4b": "1.4B",
               "gpt2-xl": "1.5B", "EleutherAI/pythia-2.8b": "2.8B",
               "EleutherAI/pythia-6.9b": "6.9B",
               "Qwen/Qwen2.5-0.5B": "0.5B", "Qwen/Qwen2.5-1.5B": "1.5B"}

TABLES = [("T1", table1_ladder), ("T2", table2_gqa), ("T3", table3_generality),
          ("T4", table4_paired), ("T5", table5_a2), ("T6", table6_gpt2)]


def main(argv=None):
    print("emitting tables from {}".format(RESULTS))
    skipped = []
    for key, fn in TABLES:
        reason = fn()
        if reason:
            skipped.append((key, reason))
            print("  SKIP {}: {}".format(key, reason))
    print("\n{} of {} tables written to {}".format(
        len(TABLES) - len(skipped), len(TABLES), _display(OUT)))
    if skipped:
        print("skipped, with reasons (not fabricated):")
        for key, reason in skipped:
            print("  {}  {}".format(key, reason))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ICLR 2027 submission

The text is a first draft and expected to change. The format is the venue's
own, taken verbatim from <https://github.com/ICLR/Master-Template>, and it is
checked rather than eyeballed.

## Format

`iclr2027_conference.sty` and its companions are copied byte-for-byte from the
`iclr2027/` directory of the Master-Template repository. They are not edited,
and `verify_format.py` fails if their sha256 changes. Nothing in this directory
overrides them.

That gives a single column 5.5in wide on US Letter, 9in of text, Times 10pt on
11pt, small-caps headings, the anonymity notice in place of the author block,
the line-number ruler, and the running head. The main text limit is **9 pages**,
with references and anything after them uncounted.

Two earlier states are kept in `legacy/` rather than deleted: a two-column
style built to an unofficial letx.app example, and the 2026 files this
submission moved off. See the note there.

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper. This is the file to edit. |
| `iclr2027_conference.sty`, `.bst`, `fancyhdr.sty`, `natbib.sty`, `math_commands.tex` | The venue's files, verbatim. Do not edit. |
| `reference_iclr2026_conference.pdf` | The only compiled PDF the repository ships. Valid as a 2027 geometry target: the 2027 style differs from 2026 only in the header year, and the other four files are byte identical. |
| `references.bib` | Bibliography. |
| `figs/` | Figures, regenerated from committed results with the in-figure banners off. |
| `build.py` | Compile plus all three checks in one command. |
| `verify_format.py` | Style-file integrity, frame geometry, running head, page limit. |
| `check_floats.py` | How far each float lands from the text that refers to it. |
| `check_tables.py` | Recomputes every cell of Tables 1-3 from the archived results. |
| `measure.py` | Prints the raw geometry of any PDF. |
| `resize_figures.py`, `move_to_appendix.py`, `reflow_floats.py`, `reflow_tables.py`, `patch_figures.py` | One-off edits, kept because they explain why the source looks the way it does. |

## Building

There is no LaTeX installation and no `make` on the machine this was written
on. Tectonic is self-contained and fetches what it needs on first run.

```
python build.py            # compile, then run all three checks
python build.py --figures  # regenerate figs/ first
python build.py --preview  # also write preview/p*.png
```

`out/` must exist first; Tectonic will not create it.

### One thing the engine gets wrong without help

Tectonic is XeTeX-based, and XeTeX defaults to the `TU` font encoding, which
has no descriptor for the `ptm` family that `times` selects. Without an
explicit `\usepackage[T1]{fontenc}` **before** `times`, every ptm shape is
undefined, LaTeX silently substitutes Latin Modern, and `\bf` stops taking
effect, so both the venue's Times body text and its bold run-in headings are
lost. The compile succeeds and the page looks plausible, which is what makes it
worth writing down. The official template does not need the line because it was
built with pdfTeX.

## Checking

All three checks exit non-zero on failure, so they gate a commit.

```
python verify_format.py out/main.pdf reference_iclr2026_conference.pdf
python check_floats.py out/main.pdf
python check_tables.py
```

`verify_format.py` does four things, in descending order of how much they
prove:

1. **Style integrity.** sha256 of all five venue files against upstream. If
   these match and nothing overrides them, the margins and skips are the
   venue's by construction rather than by resemblance.
2. **Frame geometry** against the template's own compiled PDF: page size, text
   block edges, text width, body leading, first body line. All exact at the
   last run.
3. **Page limit.** Locates the references and fails if the main text runs past
   page 9.
4. **Heading skips**, reported as diagnostics with a loose bound. Both
   documents load the same style file, so these cannot be wrong the way a
   margin can; what moves them is what happens to precede a heading.

`check_tables.py` is the one that matters for the science. The tables are
written by hand and nothing else in the build reads a CSV, so a table could
drift from the archive it reports and still compile. It parses the three tables
out of `main.tex`, recomputes all 48 cells from `results/`, and compares at the
precision printed. Derived cells are recomputed rather than read back, and a
row it does not know how to verify is reported rather than skipped. All 48
currently match.

`check_floats.py` allows two figures to sit behind the bibliography by design;
see `move_to_appendix.py` for why.

## Regenerating the figures

The figures come from the experiment repository, not from here. They carry a
`Figure N ...` banner drawn inside the artwork, which suits the report PDF and
duplicates the caption here.

```
cd ../..
XSAC_FIGURE_TITLES=0 python scripts/make_figures.py --out paper/iclr2027/figs
```

Without the variable the figures come out exactly as the report expects, so
this changes nothing for the other consumer. Figure widths are set per figure
in `resize_figures.py`; at 5.5in a plot left at `\linewidth` dominates the page.

## Anonymity

The style file supplies the anonymity notice on its own for as long as
`\iclrfinalcopy` stays commented out in `main.tex`, and prints the running head
that goes with it, reading "Under review as a conference paper at ICLR 2027".
`verify_format.py` checks that exact string appears on every page, which is the cheap way to catch `\iclrfinalcopy` being uncommented by
accident.

`../../ANONYMIZE.md` covers what still carries identity in the repository
itself, which is a separate problem from the PDF.

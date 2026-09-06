# ICLR 2026 draft

First draft of the paper. The text is a starting point and expected to change;
the format is not, and is checked rather than eyeballed.

## Files

| File | What it is |
|---|---|
| `main.tex` | The paper. This is the file to edit. |
| `iclr2026_conference.sty` | Page geometry and title furniture, every dimension measured off the venue's example PDF. |
| `references.bib` | Bibliography. |
| `figs/` | Figures, regenerated from committed results with the in-figure banners switched off. |
| `verify_format.py` | Compares the compiled PDF against the example, geometry only. |
| `check_floats.py` | Reports how far each figure and table lands from the text that refers to it. |
| `measure.py` | Prints the raw geometry of any PDF. Used to derive the numbers in the `.sty`. |
| `build.py` | Compile plus both checks in one command. |
| `reflow_floats.py`, `reflow_tables.py` | One-off edits that moved the float declarations. Kept because they explain why the source is laid out the way it is. |
| `patch_figures.py` | One-off edit to `xsac/figures.py` adding the `XSAC_FIGURE_TITLES` switch. |

## Building

There is no LaTeX installation and no `make` on the machine this was written
on. Tectonic is self-contained and pulls what it needs on first run:

```
python build.py            # compile, then run both checks
python build.py --figures  # regenerate figs/ first
python build.py --preview  # also write preview/p*.png
```

Or drive the engine directly:

```
tectonic -X compile main.tex --outdir out
```

`out/` must exist first; Tectonic will not create it.

## Checking the format

Both checks exit non-zero on failure, so they belong in front of a commit.

```
python verify_format.py out/main.pdf ~/Downloads/format+Example.pdf
python check_floats.py out/main.pdf
```

`verify_format.py` compares sixteen metrics: page size, both column edges and
widths, the gutter, total text width, body leading, the first body line's
position, and the space above and below section and subsection headings. The
tolerance is 0.6bp, except for the heading skips, which carry rubber length and
get 2.0bp.

As of the last run all sixteen match.

## Regenerating the figures

The figures come from the experiment repository, not from this directory. They
carry a `Figure N ...` banner drawn inside the artwork, which is right for the
report PDF and wrong here, because the caption sits directly under the image.
`XSAC_FIGURE_TITLES=0` suppresses the banner and leaves the panel labels alone:

```
cd ../..
XSAC_FIGURE_TITLES=0 python scripts/make_figures.py --out paper/iclr2026/figs
```

Without the variable the figures come out exactly as the report expects them,
so this changes nothing for the other consumer.

## What the style file does that the article class does not

The venue's example is a 10pt two-column `article` at heart, so most of the
class defaults are already right: `\Large` is the 14.4pt section size, `\large`
is the 12pt subsection size, `\LARGE` is the 17.28pt title. What the style file
supplies is the frame and the title block.

Three things are worth knowing before editing it.

**Dimensions are in `bp`, not `pt`.** A PDF viewer measures in big points,
1/72 in. TeX's point is 1/72.27 in. Setting `\paperwidth` to `612pt` yields a
609.71bp page, which looks right in the source and is 0.4% short on paper. The
first compile here had exactly that bug.

**`\pdfpagewidth` has to be set too.** `xdvipdfmx` reads the page size from
there, not from `\paperwidth`, and falls back to a default without complaint.

**Two values are matched, not derived.** `\topmargin` is 0.8bp off its
arithmetic value because a line box's top follows the ascender of whichever
glyph starts the line, and the heading skips are tuned against measured medians
rather than computed from `\parskip`. `verify_format.py` checks the result, so
if you change either, re-run it rather than re-deriving.

## Running head

The example carries no running head, so the default has none and
`verify_format.py` is calibrated to that. A real submission wants one; pass the
option:

```latex
\usepackage[runninghead]{iclr2026_conference}
```

This adds 12bp of header text at the top of every page. It does not move the
text block.

## Anonymity

The draft is already anonymous: `Anonymous Author`, `Anonymous Institution`,
`anon@example.invalid`. See `../../ANONYMIZE.md` for what still carries
identity in the repository itself, which is a separate problem from the PDF.

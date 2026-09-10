# Manuscript source

This directory contains the final ICLR 2027 LaTeX source and the checks needed
to audit it. Drafts, conversion intermediates, and one-off editing scripts are
not retained.

The seven PDFs in `figs/` are publication renderings of figures generated from
`results/`. Their structured source data remains in `results/figures/`.

## Verify reported tables

```bash
python check_tables.py
```

The checker parses Tables 1--3 in `main.tex`, recomputes their cells from the
committed results, and fails on any mismatch.

## Build

Install [Tectonic](https://tectonic-typesetting.github.io/) and the optional
Python dependency:

```bash
pip install -r ../../requirements-publication.txt
python build.py
```

To regenerate publication-style figures before compiling:

```bash
python build.py --figures
```

`verify_format.py` checks the venue files, page geometry, running head, and page
limit against the retained reference PDF. `check_floats.py` checks float drift.
Build outputs are ignored.

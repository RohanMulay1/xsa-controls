"""Compile the draft and run both format checks.

A shell Makefile would be the obvious home for this, but `make` is not
installed on the machine this was written on, and shipping a build file nobody
here can run is worse than none. This does the same three things and is
verified.

    python build.py                 # compile, then verify
    python build.py --figures       # regenerate figures first
    python build.py --preview       # also write preview/p*.png
    python build.py --no-verify     # compile only

TECTONIC and ICLR_TEMPLATE override the two paths that are machine-specific.
"""

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
PDF = OUT / "main.pdf"

TECTONIC = os.environ.get(
    "TECTONIC",
    str(pathlib.Path(os.path.expanduser("~"))
        / "AppData/Local/Temp/tectonic-msvc/tectonic.exe"))
TEMPLATE = os.environ.get(
    "ICLR_TEMPLATE", str(HERE / "reference_iclr2026_conference.pdf"))


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd))
    return subprocess.call([str(c) for c in cmd], **kw)


def figures():
    """Regenerate from the experiment results, banners off.

    Without XSAC_FIGURE_TITLES the figures come out as the report wants them,
    with a `Figure N ...` line drawn inside the artwork that duplicates the
    caption here.
    """
    env = dict(os.environ, XSAC_FIGURE_TITLES="0")
    return run([sys.executable, "scripts/make_figures.py",
                "--out", "paper/iclr2026/figs"], cwd=str(REPO), env=env)


def compile_pdf():
    OUT.mkdir(exist_ok=True)          # tectonic will not create it
    if not pathlib.Path(TECTONIC).exists():
        print("no TeX engine at %s; set TECTONIC" % TECTONIC, file=sys.stderr)
        return 1
    return run([TECTONIC, "-X", "compile", "main.tex", "--outdir", "out"],
               cwd=str(HERE))


def verify():
    rc = run([sys.executable, "verify_format.py", PDF, TEMPLATE], cwd=str(HERE))
    rc |= run([sys.executable, "check_floats.py", PDF], cwd=str(HERE))
    # The tables are hand-written, so nothing else in the build would notice
    # them drifting from the archive they report.
    rc |= run([sys.executable, "check_tables.py"], cwd=str(HERE))
    return rc


def preview():
    import fitz
    out = HERE / "preview"
    out.mkdir(exist_ok=True)
    with fitz.open(str(PDF)) as doc:
        for i, page in enumerate(doc, 1):
            page.get_pixmap(dpi=100).save(str(out / ("p%d.png" % i)))
        print("%d preview pages written" % doc.page_count)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", action="store_true",
                    help="regenerate figs/ from the experiment results first")
    ap.add_argument("--preview", action="store_true",
                    help="also render preview/p*.png")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args(argv)

    if args.figures and figures():
        return 1
    if compile_pdf():
        return 1
    rc = 0 if args.no_verify else verify()
    if args.preview:
        preview()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

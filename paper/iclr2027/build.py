"""Build and verify the retained ICLR manuscript."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
PDF = OUT / "main.pdf"
TEMPLATE = pathlib.Path(os.environ.get(
    "ICLR_TEMPLATE", HERE / "reference_iclr2026_conference.pdf"
))


def run(command, **kwargs):
    print("$", " ".join(str(part) for part in command))
    return subprocess.call([str(part) for part in command], **kwargs)


def regenerate_figures():
    env = dict(os.environ, XSAC_FIGURE_TITLES="0")
    return run(
        [sys.executable, "scripts/make_figures.py", "--out", "paper/iclr2027/figs"],
        cwd=REPO,
        env=env,
    )


def compile_pdf(tectonic):
    OUT.mkdir(exist_ok=True)
    return run(
        [tectonic, "-X", "compile", "main.tex", "--outdir", "out"],
        cwd=HERE,
    )


def verify():
    commands = (
        [sys.executable, "verify_format.py", PDF, TEMPLATE],
        [sys.executable, "check_floats.py", PDF],
        [sys.executable, "check_tables.py"],
    )
    return_code = 0
    for command in commands:
        return_code |= run(command, cwd=HERE)
    return return_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--tectonic",
        default=os.environ.get("TECTONIC") or shutil.which("tectonic"),
    )
    args = parser.parse_args(argv)

    if args.figures and regenerate_figures():
        return 1
    if not args.tectonic:
        parser.error("Tectonic was not found; install it or pass --tectonic")
    if compile_pdf(args.tectonic):
        return 1
    return 0 if args.no_verify else verify()


if __name__ == "__main__":
    raise SystemExit(main())

"""Assert every figure survives greyscale, print and colour-blind viewing.

Spec section 11 forbids colour as the only thing separating two series, and
nothing enforced it. This does, on two levels:

1. **The palette**, checked directly. Every pair of series colours must differ
   in relative luminance by at least ``MIN_LUMINANCE_GAP``, and every series
   must also carry a distinct marker and linestyle so identity survives even
   when luminance does not.
2. **The rendered PNGs**, checked as pixels. Each figure is converted to
   luminance and must retain real tonal spread: a plot whose series collapse
   to one grey is unreadable in print no matter what the palette says.

    python scripts/check_figures.py

Exits non-zero on any failure, so CI fails rather than shipping a figure a
reviewer cannot read. Run it after make_figures.py.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from xsac.figures import LINESTYLES, MARKERS, SERIES  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"

#: Minimum difference in relative luminance between any two series, on 0-1.
#: 0.10 is roughly the point at which two greys stay separable in print at
#: figure line widths.
MIN_LUMINANCE_GAP = 0.10

#: A rendered figure must use at least this much of the tonal range, measured
#: as the interquartile spread of its non-background luminance.
MIN_TONAL_SPREAD = 0.05


def _srgb_to_linear(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def relative_luminance(rgb):
    """WCAG relative luminance from 8-bit sRGB."""
    r, g, b = (_srgb_to_linear(np.asarray(x, dtype=float)) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def check_palette():
    """Series must be separable without colour."""
    failures = []
    lum = {k: relative_luminance(hex_to_rgb(v)) for k, v in SERIES.items()}
    names = sorted(SERIES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = abs(lum[a] - lum[b])
            if gap < MIN_LUMINANCE_GAP:
                failures.append(
                    "{} and {} differ by only {:.3f} in luminance "
                    "(need {:.2f}); in greyscale they are the same line"
                    .format(a, b, gap, MIN_LUMINANCE_GAP))
    # Redundant encoding: even where luminance is close, marker and linestyle
    # must still tell the series apart.
    for label, mapping in (("marker", MARKERS), ("linestyle", LINESTYLES)):
        missing = [k for k in SERIES if k not in mapping]
        if missing:
            failures.append("no {} defined for {}".format(
                label, ", ".join(sorted(missing))))
        used = [mapping[k] for k in SERIES if k in mapping]
        if len(set(used)) != len(used):
            failures.append("{}s are not unique across series: {}".format(
                label, used))
    print("  palette: {} series, luminance {:.3f}-{:.3f}".format(
        len(SERIES), min(lum.values()), max(lum.values())))
    return failures


def check_rendered(path):
    """A rendered figure must keep tonal structure once desaturated."""
    try:
        from PIL import Image
    except ImportError:
        return ["Pillow not installed; cannot check rendered figures"]
    img = Image.open(path).convert("RGB")
    a = np.asarray(img, dtype=float)
    lin = _srgb_to_linear(a)
    lum = (0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2])
    ink = lum[lum < 0.95]                    # drop the white background
    if ink.size < 100:
        return ["{}: almost every pixel is background".format(path.name)]
    spread = float(np.percentile(ink, 90) - np.percentile(ink, 10))
    if spread < MIN_TONAL_SPREAD:
        return ["{}: luminance spread {:.3f} is below {:.2f}; the series "
                "collapse to one grey in print".format(
                    path.name, spread, MIN_TONAL_SPREAD)]
    print("  {:<28} tonal spread {:.3f}  ink pixels {}".format(
        path.name, spread, ink.size))
    return []


def main(argv=None):
    print("checking figure accessibility")
    failures = list(check_palette())

    pngs = sorted(FIGURES.glob("*.png")) if FIGURES.exists() else []
    if not pngs:
        print("  no rendered figures found in {}; run make_figures.py first"
              .format(FIGURES.relative_to(ROOT)))
    for png in pngs:
        failures.extend(check_rendered(png))

    print()
    if failures:
        print("FAILED: {} problem(s)".format(len(failures)))
        for f in failures:
            print("  - {}".format(f))
        return 1
    print("PASS: palette and {} rendered figure(s) survive greyscale".format(
        len(pngs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

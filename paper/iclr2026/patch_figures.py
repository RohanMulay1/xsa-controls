"""One-off edit to xsac/figures.py: make the baked-in figure titles optional.

The report PDF wants a title inside each figure, because a reader scrolling
past it may never reach the caption. A paper does not: the caption sits
directly under the image, and a second "Figure 1 ..." line drawn inside the
artwork reads as a duplicate and pushes the axes down.

So the three `fig.suptitle("Figure N ...")` calls become `_suptitle`, which is
a no-op when XSAC_FIGURE_TITLES=0. Nothing else changes, and the default keeps
the report's figures exactly as they were.
"""

import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[2] / "xsac" / "figures.py"

HELPER = '''

#: Figures carry their own "Figure N ..." banner in the report PDF, where the
#: caption may be several screens away. In a paper the caption sits under the
#: image and the banner duplicates it, so set XSAC_FIGURE_TITLES=0 to drop it.
EMBED_FIGURE_TITLES = os.environ.get("XSAC_FIGURE_TITLES", "1") != "0"


def _suptitle(fig, text, **kwargs):
    """Draw the in-figure banner, unless it has been switched off."""
    if EMBED_FIGURE_TITLES:
        fig.suptitle(text, **kwargs)
'''


def main():
    text = SRC.read_text(encoding="utf-8")
    if "EMBED_FIGURE_TITLES" in text:
        print("already patched")
        return 0

    if "\nimport os\n" not in text:
        text = text.replace("\nimport math\n", "\nimport math\nimport os\n", 1)

    anchor = "class FigureSkipped(Exception):"
    assert anchor in text, "anchor for the helper not found"
    text = text.replace(anchor, HELPER.strip() + "\n\n\n" + anchor, 1)

    n = len(re.findall(r"\bfig\.suptitle\(", text))
    text = re.sub(r"\bfig\.suptitle\(", "_suptitle(fig, ", text)
    SRC.write_text(text, encoding="utf-8")
    print("patched %d suptitle calls in %s" % (n, SRC))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

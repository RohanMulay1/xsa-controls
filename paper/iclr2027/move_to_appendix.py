"""Move two supporting figures behind the bibliography.

ICLR 2026 caps the main text at nine pages, with references and material after
them uncounted. The draft ran to ten. Rather than cut an argument, the two
figures whose content is fully stated in the prose that cites them move to the
appendix: the power curve, whose two eight-seed values appear in the text, and
the generality panel, whose four numbers appear in the paragraph beside it.

Cross-references keep working; the figures simply appear later. Nothing is
deleted and no claim moves.

One-off. Kept so the ordering in main.tex is explicable.
"""

import pathlib
import re

MOVE = ["fig7_power", "fig4_generality"]


def main():
    path = pathlib.Path(__file__).with_name("main.tex")
    text = path.read_text(encoding="utf-8")

    moved = []
    for stem in MOVE:
        pattern = re.compile(
            r"\\begin\{figure\}\[tb\](?:(?!\\begin\{figure\}).)*?"
            r"figs/%s\.pdf.*?\\end\{figure\}\n\n" % re.escape(stem), re.S)
        m = pattern.search(text)
        if not m:
            print("  warning: %s not found, or already moved" % stem)
            continue
        moved.append(m.group(0))
        text = text[:m.start()] + text[m.end():]

    if not moved:
        return 1

    anchor = "\\appendix\n"
    assert anchor in text, "no \\appendix in main.tex"
    intro = ("\n% Moved here to keep the main text within the nine-page limit.\n"
             "% Both figures' numbers are stated in the sections that cite\n"
             "% them, so nothing is lost from the argument by their position.\n"
             "% See move_to_appendix.py.\n")
    text = text.replace(anchor, anchor + intro + "".join(moved), 1)
    path.write_text(text, encoding="utf-8")
    print("moved %d figures behind the bibliography" % len(moved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

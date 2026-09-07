"""Move every figure declaration to the top of the Results section.

LaTeX places a float on the earliest page at or after the one it is declared
on. Declaring each figure next to the paragraph that discusses it therefore
guarantees it can only ever appear later, and with ten floats the deferral
compounds: figures were surfacing two pages after their first reference.
Declaring them all at the start of Results, in reference order, lets the
placement algorithm fill from that page onward.

One-off. Kept in the tree because it documents why the figures sit where they
do in the source, which otherwise looks like carelessness.
"""

import pathlib
import re

ORDER = ["fig6_a2_scatter", "fig2_paired_delta", "fig1_gates", "fig7_power",
         "fig3_ladder", "fig4_generality", "fig5_gqa"]

PATTERN = re.compile(r"\\begin\{figure\*?\}\[tb\].*?\\end\{figure\*?\}\n\n",
                     re.S)


def main():
    path = pathlib.Path(__file__).with_name("main.tex")
    text = path.read_text(encoding="utf-8")

    blocks = PATTERN.findall(text)
    if not blocks:
        print("no figure blocks found; already moved?")
        return 1
    for block in blocks:
        text = text.replace(block, "")

    def rank(block):
        for i, name in enumerate(ORDER):
            if name in block:
                return i
        return len(ORDER)

    blocks.sort(key=rank)
    anchor = "\\section{Results}\n"
    assert anchor in text, "Results section not found"
    text = text.replace(anchor, anchor + "\n" + "".join(blocks), 1)
    path.write_text(text, encoding="utf-8")
    print("moved %d figures:" % len(blocks),
          ", ".join(next((n for n in ORDER if n in b), "?") for b in blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

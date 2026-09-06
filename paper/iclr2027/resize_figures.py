"""Resize the figures for the venue's single-column format.

The draft was previously set in two columns, where a figure at \\columnwidth
was 245bp wide. ICLR is a single 5.5in column, so \\columnwidth is now 396bp:
the same directive would render every plot half again as large and push the
paper over its page limit for no gain in legibility.

Widths below are set per figure from its native aspect ratio, so a wide panel
strip still spans the text block while a single square panel does not.

One-off. Kept in the tree so the widths in main.tex do not look arbitrary.
"""

import pathlib
import re

# Native sizes are in the PDFs themselves; these fractions were chosen so a
# figure's rendered height stays near a third of the text block.
WIDTHS = {
    "fig6_a2_scatter": "0.98\\linewidth",   # 849x316, a three-panel strip
    "fig1_gates": "0.55\\linewidth",        # 471x266, two panels
    "fig2_paired_delta": "0.42\\linewidth",  # 347x291, one sparse panel
    "fig3_ladder": "0.50\\linewidth",       # 463x317
    "fig4_generality": "0.48\\linewidth",   # 442x328
    "fig5_gqa": "0.48\\linewidth",          # 410x342
    "fig7_power": "0.50\\linewidth",        # 420x293
}


def main():
    path = pathlib.Path(__file__).with_name("main.tex")
    text = path.read_text(encoding="utf-8")

    # A single-column document has no \begin{figure*}.
    text = text.replace("\\begin{figure*}[t]", "\\begin{figure}[tb]")
    text = text.replace("\\end{figure*}", "\\end{figure}")

    changed = 0
    for stem, width in WIDTHS.items():
        pattern = re.compile(
            r"\\includegraphics\[width=[^\]]*\]\{figs/%s\.pdf\}" % re.escape(stem))
        new = "\\includegraphics[width=%s]{figs/%s.pdf}" % (width, stem)
        # A plain string replacement: re.sub would read the backslashes in
        # `new` as escape sequences in the template.
        text, n = pattern.subn(lambda _m, r=new: r, text)
        if n:
            changed += n
        else:
            print("  warning: no \\includegraphics found for %s" % stem)

    path.write_text(text, encoding="utf-8")
    print("set the width of %d figures for a single-column text block" % changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

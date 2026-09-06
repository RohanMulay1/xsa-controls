"""Move table declarations to the top of Results, for the same reason as the
figures: a float cannot land before the page it is declared on, and Table 3
was arriving two pages after the text that discusses it.

See reflow_floats.py. Run once; check_floats.py measures the result.
"""

import pathlib
import re

ORDER = ["tab:reliability", "tab:a2", "tab:paired"]
PATTERN = re.compile(r"\\begin\{table\}\[tb\].*?\\end\{table\}\n\n", re.S)


def main():
    path = pathlib.Path(__file__).with_name("main.tex")
    text = path.read_text(encoding="utf-8")

    blocks = PATTERN.findall(text)
    if not blocks:
        print("no table blocks found; already moved?")
        return 1
    for block in blocks:
        text = text.replace(block, "")

    def rank(block):
        for i, label in enumerate(ORDER):
            if label in block:
                return i
        return len(ORDER)

    blocks.sort(key=rank)
    anchor = "\\section{Results}\n"
    assert anchor in text, "Results section not found"
    text = text.replace(anchor, anchor + "\n" + "".join(blocks), 1)
    path.write_text(text, encoding="utf-8")
    print("moved %d tables:" % len(blocks),
          ", ".join(next((n for n in ORDER if n in b), "?") for b in blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

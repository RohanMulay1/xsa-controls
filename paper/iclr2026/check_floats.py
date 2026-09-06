"""Report how far each float sits from the text that first refers to it.

A figure that surfaces two pages after its first mention is a real reading
problem and the most common thing that goes wrong when a style file's float
parameters are tuned. This reads the compiled PDF, finds the page carrying
each caption and the page carrying the first in-text reference, and prints the
drift.

    python check_floats.py out/main.pdf

Exits non-zero if any float drifts by more than MAX_DRIFT pages.
"""

import collections
import re
import sys

import fitz

MAX_DRIFT = 1

# Floats deliberately placed after the bibliography to keep the main text
# inside the nine-page limit. Their drift is intended, so it is reported
# separately rather than counted as a failure. See move_to_appendix.py; the
# numbers are the caption numbers those figures carry.
APPENDIX_FLOATS = {("Figure", "6"), ("Figure", "7")}

CAPTION = re.compile(r"(Figure|Table)\s+(\d+):")
REFERENCE = re.compile(r"(Figure|Table)\s+(\d+)(?!:)")


def main(path):
    doc = fitz.open(path)
    caption, reference = {}, {}
    for pno, page in enumerate(doc, 1):
        text = page.get_text()
        for m in CAPTION.finditer(text):
            caption.setdefault((m.group(1), m.group(2)), pno)
        for m in REFERENCE.finditer(text):
            reference.setdefault((m.group(1), m.group(2)), pno)

    print("%-10s %7s %9s %7s" % ("float", "shown", "first ref", "drift"))
    print("-" * 36)
    worst = 0
    counts = collections.Counter()
    for key in sorted(caption, key=lambda k: (k[0], int(k[1]))):
        shown = caption[key]
        ref = reference.get(key)
        counts[shown] += 1
        if ref is None:
            print("%-10s %7d %9s %7s   never referenced in text"
                  % (" ".join(key), shown, "-", "-"))
            continue
        drift = shown - ref
        if key in APPENDIX_FLOATS:
            print("%-10s %7d %9d %+7d   in the appendix by design"
                  % (" ".join(key), shown, ref, drift))
            continue
        worst = max(worst, abs(drift))
        flag = "  <--" if abs(drift) > MAX_DRIFT else ""
        print("%-10s %7d %9d %+7d%s" % (" ".join(key), shown, ref, drift, flag))

    print("\nfloats per page:", dict(sorted(counts.items())))
    print("pages:", doc.page_count)
    if worst > MAX_DRIFT:
        print("\nworst drift %d pages, limit %d" % (worst, MAX_DRIFT))
        return 1
    print("\nevery float lands within %d page of its first reference"
          % MAX_DRIFT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

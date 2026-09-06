"""Compare the compiled draft against the venue's example PDF, geometry only.

The two documents share no words, so nothing here reads text. It measures the
frame: page size, where each column starts and stops, the gutter, the body
leading, and the set of font sizes in use. Every number is taken from the
rendered PDF with PyMuPDF, which reports coordinates in big points, the same
unit the style file now uses.

    python verify_format.py out/main.pdf ~/Downloads/format+Example.pdf

Exit status is non-zero if any metric differs by more than TOL, so this can sit
in front of a commit.
"""

import collections
import statistics
import sys

import fitz

# 0.6bp is under a hairline rule and well below what a reader could see.
# Anything larger is a genuine difference in the style file rather than
# rounding in the glyph bounding boxes.
TOL = 0.6

# Heading skips carry rubber length, so a page that had to stretch or shrink
# to fill its column moves them by a point or so. The median is compared
# against a looser bound for that reason; it is still far tighter than the
# 6pt of \parskip that would show if a skip were actually wrong.
LOOSE = {"sec_above": 2.0, "sec_below": 2.0,
         "subsec_above": 2.0, "subsec_below": 2.0}

BODY_MIN, BODY_MAX = 9.6, 10.4


def body_lines(doc):
    """Lines set at the body size, which are the ones that define the frame.

    Page 1 is excluded: its upper half is the title block and a full-width
    abstract, both of which legitimately ignore the column grid.
    """
    out = []
    for pno, page in enumerate(doc):
        if pno == 0:
            continue
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                if not spans:
                    continue
                if not "".join(s["text"] for s in spans).strip():
                    continue
                if BODY_MIN <= spans[0]["size"] <= BODY_MAX:
                    out.append((pno, line["bbox"]))
    return out


def column_edges(lines):
    """Left edges cluster hard; right edges do not, because the last line of
    every paragraph is short. So the left edges are taken as the two modes,
    and each right edge as the mode of the justified lines in that column."""
    lefts = collections.Counter(round(b[0]) for _, b in lines)
    c1, c2 = sorted(x for x, _ in lefts.most_common(2))
    mid = (c1 + c2) / 2.0
    edges = []
    for lo, hi in ((c1 - 2, mid), (c2 - 2, 10 ** 6)):
        rights = collections.Counter(
            round(b[2]) for _, b in lines if lo <= b[0] < hi)
        # The justified right edge is the most common, not the largest: a
        # full-width float caption would otherwise win.
        edges.append(max(x for x, n in rights.items() if n >= 0.15 * sum(
            rights.values())))
    return float(c1), float(edges[0]), float(c2), float(edges[1])


def leading(lines):
    """Baseline-to-baseline distance, measured inside one column on one page.

    Measuring across merged columns produces meaningless small deltas, because
    the two columns interleave in reading order.
    """
    by_col = collections.defaultdict(list)
    c1, _, c2, _ = column_edges(lines)
    for pno, b in lines:
        key = (pno, 0 if b[0] < (c1 + c2) / 2 else 1)
        by_col[key].append(round(b[1], 2))
    deltas = []
    for ys in by_col.values():
        ys = sorted(ys)
        deltas += [round(b - a, 2) for a, b in zip(ys, ys[1:]) if 8 < b - a < 20]
    return statistics.mode(deltas) if deltas else None


def frame(doc):
    lines = body_lines(doc)
    c1l, c1r, c2l, c2r = column_edges(lines)
    return {
        "page_width": round(doc[0].rect.width, 1),
        "page_height": round(doc[0].rect.height, 1),
        "col1_left": c1l,
        "col1_right": c1r,
        "col1_width": c1r - c1l,
        "col2_left": c2l,
        "col2_right": c2r,
        "col2_width": c2r - c2l,
        "gutter": c2l - c1r,
        "textwidth": c2r - c1l,
        "body_leading": leading(lines),
        "text_top": min(round(b[1], 1) for _, b in lines),
        "sec_above": heading_gaps(doc, SECTION_SIZE)[0],
        "sec_below": heading_gaps(doc, SECTION_SIZE)[1],
        "subsec_above": heading_gaps(doc, SUBSECTION_SIZE)[0],
        "subsec_below": heading_gaps(doc, SUBSECTION_SIZE)[1],
    }


SECTION_SIZE, SUBSECTION_SIZE = 14.3, 12.0
BODY_SIZES = (9.9, 10.0, 10.1)


def _page_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            text = "".join(s["text"] for s in spans).strip()
            if text:
                out.append((round(line["bbox"][0], 1), round(line["bbox"][1], 1),
                            round(spans[0]["size"], 1)))
    return out


def heading_gaps(doc, size):
    """Space above and below headings, measured between line-box tops.

    Only headings with a body line on the relevant side are counted, so a
    heading at the top of a column contributes nothing to the `above' figure
    and a heading followed by a float contributes nothing to `below'.
    """
    above, below = [], []
    for page in doc:
        rows = _page_lines(page)
        for col in (0, 1):
            sel = sorted([r for r in rows if (r[0] < 300) == (col == 0)],
                         key=lambda r: r[1])
            for i, (_, y, sz) in enumerate(sel):
                if sz != size:
                    continue
                if i > 0 and sel[i - 1][2] in BODY_SIZES:
                    above.append(y - sel[i - 1][1])
                if i + 1 < len(sel) and sel[i + 1][2] in BODY_SIZES:
                    below.append(sel[i + 1][1] - y)
    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    return med(above), med(below)


def size_inventory(doc):
    sizes = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    if s["text"].strip():
                        sizes[round(s["size"], 1)] += 1
    return sizes


def main(draft_path, template_path):
    draft, template = fitz.open(draft_path), fitz.open(template_path)
    fa, fb = frame(draft), frame(template)

    print("%-14s %10s %10s %8s" % ("metric", "draft", "template", "delta"))
    print("-" * 46)
    bad = []
    for key in fa:
        va, vb = fa[key], fb[key]
        if va is None or vb is None:
            print("%-14s %10s %10s" % (key, va, vb))
            continue
        d = va - vb
        flag = "  <-- MISMATCH" if abs(d) > LOOSE.get(key, TOL) else ""
        if flag:
            bad.append(key)
        print("%-14s %10.1f %10.1f %8.1f%s" % (key, va, vb, d, flag))

    print("\nfont sizes in use (bp: line count)")
    sa, sb = size_inventory(draft), size_inventory(template)
    for sz in sorted(set(sa) | set(sb)):
        print("   %5.1f   draft %-5d template %-5d" % (sz, sa.get(sz, 0),
                                                       sb.get(sz, 0)))

    print("\ndraft %d pages, template %d pages" % (draft.page_count,
                                                   template.page_count))
    if bad:
        print("\n%d mismatches beyond %.1fbp: %s" % (len(bad), TOL,
                                                     ", ".join(bad)))
        return 1
    print("\ngeometry matches the template within %.1fbp on every metric" % TOL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

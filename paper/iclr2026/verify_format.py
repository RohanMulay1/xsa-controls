"""Compare the compiled draft against the venue's own compiled template.

The reference is `reference_iclr2026_conference.pdf`, the PDF shipped in
https://github.com/ICLR/Master-Template alongside the style file this document
loads unmodified. Comparing against it catches the case that matters: a
preamble option or a stray package that moves the frame even though the
official `.sty` is present and untouched.

    python verify_format.py out/main.pdf reference_iclr2026_conference.pdf

ICLR 2026 is a single column, so the metrics are the text block rather than
two columns and a gutter. Nothing here reads words; the two documents share no
content.

Exit status is non-zero if any metric differs by more than its tolerance.
"""

import collections
import statistics
import sys

import fitz

# 0.6bp is below a hairline and well under what a reader could see. Anything
# larger means the frame moved.
TOL = 0.6

# Heading skips are diagnostics, not guarantees. Both documents load the same
# unmodified style file, so a skip cannot be wrong here in the way a margin
# can; what moves the median is what happens to precede a heading, which is a
# property of the writing. The style file being byte-identical is checked
# directly instead, below. These bounds catch a gross regression and nothing
# finer.
LOOSE = {"sec_above": 6.0, "sec_below": 6.0,
         "subsec_above": 6.0, "subsec_below": 6.0}

# sha256 of the files taken verbatim from
# https://github.com/ICLR/Master-Template, iclr2026/. If one of these changes,
# the format is no longer the venue's whatever the rendered page looks like.
UPSTREAM = {
    "iclr2026_conference.sty":
        "a4852f68e080d6c5245057ca2039100b409e31727898aa93c03d78ddb84374a3",
    "iclr2026_conference.bst":
        "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
    "fancyhdr.sty":
        "b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea",
    "natbib.sty":
        "88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d",
    "math_commands.tex":
        "90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9",
}

BODY_MIN, BODY_MAX = 9.6, 10.4
SECTION_SIZE, SUBSECTION_SIZE = 12.0, 10.0
BODY_SIZES = (9.9, 10.0, 10.1)

# "There will be a strict upper limit of 9 pages for the main text of the
# initial submission, with unlimited additional pages for citations."
# iclr2026_conference.tex, line 131.
MAIN_TEXT_PAGE_LIMIT = 9

# The submission style prints a vertical line-number ruler in the left margin
# and a running head at the top. Both are set well outside the text block, so
# body-size lines are filtered by x as well as by size.
TEXT_LEFT_MIN = 90.0


def body_lines(doc):
    """Lines set at the body size inside the text block.

    Page 1 is excluded: it carries the title, the anonymity notice and the
    indented abstract, none of which sit on the body measure.
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
                if not (BODY_MIN <= spans[0]["size"] <= BODY_MAX):
                    continue
                if line["bbox"][0] < TEXT_LEFT_MIN:
                    continue          # margin ruler, not text
                out.append((pno, line["bbox"]))
    return out


def text_edges(lines):
    """Left edge is the mode. The right edge is the mode of the justified
    lines, not the maximum: the last line of every paragraph is short, and a
    single over-wide box would otherwise define the measure."""
    lefts = collections.Counter(round(b[0]) for _, b in lines)
    left = lefts.most_common(1)[0][0]
    rights = collections.Counter(round(b[2]) for _, b in lines)
    total = sum(rights.values())
    right = max(x for x, n in rights.items() if n >= 0.15 * total)
    return float(left), float(right)


def leading(lines):
    """Baseline-to-baseline distance within a page."""
    by_page = collections.defaultdict(list)
    for pno, b in lines:
        by_page[pno].append(round(b[1], 2))
    deltas = []
    for ys in by_page.values():
        ys = sorted(ys)
        deltas += [round(b - a, 2) for a, b in zip(ys, ys[1:]) if 8 < b - a < 20]
    return statistics.mode(deltas) if deltas else None


def _page_lines(page):
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = line["spans"]
            text = "".join(s["text"] for s in spans).strip()
            # The running head is body-size text inside the text block, so
            # without this it becomes the neighbour of the first heading on
            # every page and reports a skip of half the top margin.
            if text and line["bbox"][0] >= TEXT_LEFT_MIN \
                    and not text.startswith(RUNNING_HEAD):
                out.append((round(line["bbox"][1], 1),
                            round(spans[0]["size"], 1), text))
    return out


RUNNING_HEAD = "Under review as a conference paper"


def _is_smallcaps_heading(text):
    """The style sets headings in small caps, which reach the PDF as capitals
    in the regular face. Weight cannot identify them, so case and length do.

    The lower bound on length matters: a notation table of single capitals,
    like the one in the venue's own template, otherwise reads as dozens of
    subsection headings and drags the measured skip down by a third.
    """
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 6 or len(text) >= 70:
        return False
    return all(c.isupper() for c in letters)


def heading_gaps(doc, size):
    """Space above and below headings, between line-box tops.

    A heading counts only where a body line sits on the relevant side, so one
    at the top of a page contributes nothing above and one followed by a float
    contributes nothing below. Section headings are 12pt; subsection headings
    share the 10pt body size, so those are picked out by their capitals.
    """
    above, below = [], []
    for page in doc:
        rows = sorted(_page_lines(page))
        for i, (y, sz, text) in enumerate(rows):
            if sz != size:
                continue
            if size in BODY_SIZES and not _is_smallcaps_heading(text):
                continue
            # A numbered heading reaches the extractor as two runs on one
            # baseline, the number and the text. Neighbours sharing this y are
            # the same visual line, not the paragraph above or below it.
            prev = next((r for r in reversed(rows[:i]) if y - r[0] > 1), None)
            nxt = next((r for r in rows[i + 1:] if r[0] - y > 1), None)
            if prev and prev[1] in BODY_SIZES \
                    and not _is_smallcaps_heading(prev[2]):
                above.append(y - prev[0])
            if nxt and nxt[1] in BODY_SIZES \
                    and not _is_smallcaps_heading(nxt[2]):
                below.append(nxt[0] - y)
    med = (lambda xs: round(statistics.median(xs), 1) if xs else None)
    return med(above), med(below)


def frame(doc):
    lines = body_lines(doc)
    left, right = text_edges(lines)
    sec_a, sec_b = heading_gaps(doc, SECTION_SIZE)
    sub_a, sub_b = heading_gaps(doc, SUBSECTION_SIZE)
    return {
        "page_width": round(doc[0].rect.width, 1),
        "page_height": round(doc[0].rect.height, 1),
        "text_left": left,
        "text_right": right,
        "text_width": right - left,
        "body_leading": leading(lines),
        "text_top": min(round(b[1], 1) for _, b in lines),
        "sec_above": sec_a,
        "sec_below": sec_b,
        "subsec_above": sub_a,
        "subsec_below": sub_b,
    }


def running_head(doc):
    """The submission style prints a running head on every page. Its absence
    means \\iclrfinalcopy was left uncommented, which de-anonymises the PDF."""
    hits = 0
    for page in doc:
        top = page.get_text("text", clip=fitz.Rect(0, 0, page.rect.width, 70))
        if "Under review as a conference paper" in top:
            hits += 1
    return hits


def size_inventory(doc):
    sizes = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    if s["text"].strip():
                        sizes[round(s["size"], 1)] += 1
    return sizes


def style_integrity():
    """The strongest check available: the format is the venue's if the files
    that define it are the venue's, byte for byte."""
    import hashlib
    import pathlib
    here = pathlib.Path(__file__).resolve().parent
    problems = []
    print("official template files")
    for name, want in sorted(UPSTREAM.items()):
        path = here / name
        if not path.exists():
            problems.append("%s is missing" % name)
            print("   MISSING   %s" % name)
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got == want:
            print("   verbatim  %-28s %s" % (name, got[:12]))
        else:
            problems.append("%s differs from upstream" % name)
            print("   MODIFIED  %-28s %s (want %s)"
                  % (name, got[:12], want[:12]))
    return problems


def main(draft_path, template_path):
    draft, template = fitz.open(draft_path), fitz.open(template_path)
    fa, fb = frame(draft), frame(template)
    style_problems = style_integrity()
    print()

    print("%-14s %10s %10s %8s" % ("metric", "draft", "official", "delta"))
    print("-" * 46)
    bad = list(style_problems)
    for key in fa:
        va, vb = fa[key], fb[key]
        if va is None or vb is None:
            print("%-14s %10s %10s   not measurable" % (key, va, vb))
            continue
        d = va - vb
        flag = "  <-- MISMATCH" if abs(d) > LOOSE.get(key, TOL) else ""
        if flag:
            bad.append(key)
        print("%-14s %10.1f %10.1f %8.1f%s" % (key, va, vb, d, flag))

    # The page limit is a formatting rule, so it is checked here rather than
    # left to be noticed. References and anything after them are uncounted.
    refs_page = None
    for pno, page in enumerate(draft, 1):
        if "REFERENCES" in page.get_text().upper():
            refs_page = pno
            break
    if refs_page is None:
        print("\ncould not locate the references; page limit unchecked")
    else:
        over = refs_page > MAIN_TEXT_PAGE_LIMIT
        print("\nmain text ends on page %d of a %d page limit "
              "(references start there; %d pages total)%s"
              % (refs_page, MAIN_TEXT_PAGE_LIMIT, draft.page_count,
                 "  <-- OVER LIMIT" if over else ""))
        if over:
            bad.append("page_limit")

    ra, rb = running_head(draft), running_head(template)
    ok = ra == draft.page_count
    print("\nrunning head on %d of %d draft pages (official: %d of %d)%s"
          % (ra, draft.page_count, rb, template.page_count,
             "" if ok else "  <-- MISMATCH"))
    if not ok:
        bad.append("running_head")

    print("\nfont sizes in use (bp: line count)")
    sa, sb = size_inventory(draft), size_inventory(template)
    for sz in sorted(set(sa) | set(sb)):
        print("   %5.1f   draft %-5d official %-5d" % (sz, sa.get(sz, 0),
                                                       sb.get(sz, 0)))

    print("\ndraft %d pages, official template %d pages"
          % (draft.page_count, template.page_count))
    if bad:
        print("\n%d mismatches: %s" % (len(bad), ", ".join(bad)))
        return 1
    print("\nframe matches the official template on every metric")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))

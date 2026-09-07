"""Render the .docx through Word and measure what Word actually produced.

Everything verify_docx.py checks is structural: the styles say Times 10pt, the
section says 5.5in of measure. Only a render says what a reader sees. Word is
installed here, so this drives it over COM, exports a PDF, and measures that
PDF the same way verify_format.py measures the LaTeX one.

    python render_docx.py                    # render, then measure
    python render_docx.py --keep-open        # leave Word running

Word is started invisibly and closed again. If a copy of Word is already
running with the file open, close it first: COM will attach to that instance.
"""

import argparse
import collections
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
DOCX = HERE / "docx" / "main.docx"
RENDERED = HERE / "docx" / "main_from_word.pdf"
LATEX_PDF = HERE / "out" / "main.pdf"

WD_EXPORT_PDF = 17
TOL_IN = 0.02
TOL_RIGHT_IN = 0.03   # glyph ink can cross the advance width slightly

# What the LaTeX PDF measures, in inches. Word will not match its line breaks,
# but it must match its frame.
EXPECTED = {
    "page_width": 8.5,
    "page_height": 11.0,
    "text_left": 1.5,
}

# Word sets ragged-right, so the longest line lands wherever the text happens
# to break: it never crosses the margin but often stops short of it, and by a
# different amount after any edit. Testing the maximum against 7.0in judged
# the wording, not the frame. The property that matters is one-sided.
RIGHT_MARGIN_IN = 7.0
RIGHT_OVERFLOW_IN = 0.03   # glyph ink may cross its advance width slightly


def render():
    import win32com.client
    if not DOCX.exists():
        print("no %s; run build_docx.py first" % DOCX, file=sys.stderr)
        return None
    if RENDERED.exists():
        RENDERED.unlink()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    try:
        doc = word.Documents.Open(str(DOCX), ReadOnly=True)
        try:
            # Fields are inserted, not evaluated, by the writer; the page
            # number in the footer is blank until Word computes it.
            doc.Fields.Update()
            doc.Repaginate()
            pages = doc.ComputeStatistics(2)   # wdStatisticPages
            doc.ExportAsFixedFormat(str(RENDERED), WD_EXPORT_PDF)
        finally:
            doc.Close(False)
    finally:
        word.Quit()
    print("Word reports %d pages; exported %s" % (pages, RENDERED.name))
    return pages


def body_lines(doc, skip_first=True, drop_trailing_space=False):
    """Lines at the body size, inside the text block.

    `drop_trailing_space` excludes lines whose text ends in whitespace. Word
    sets ragged-right, and a line that ends with a space has that space's
    advance width in its bounding box: the box crosses the right margin while
    nothing is drawn there. Measuring the margin without this filter reported
    the text block 0.04in too wide on 182 of 476 lines, 180 of which ended in
    a space.
    """
    out = []
    for pno, page in enumerate(doc):
        if skip_first and pno == 0:
            continue
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                if not spans:
                    continue
                raw = "".join(s["text"] for s in spans)
                if not raw.strip():
                    continue
                if not (9.5 <= spans[0]["size"] <= 10.5):
                    continue
                if raw.strip().startswith("Under review as a conference"):
                    continue
                if drop_trailing_space and raw != raw.rstrip():
                    continue
                out.append(line["bbox"])
    return out


def body_lines_by_page(doc):
    """Same selection, grouped by page, for measuring leading.

    Leading has to be measured inside a page. Taking consecutive differences
    over every line in the document puts a page break between two of them and
    the resulting number means nothing; that is where the 9.4pt reading came
    from.
    """
    pages = collections.defaultdict(list)
    for pno, page in enumerate(doc):
        if pno == 0:
            continue
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                if not spans:
                    continue
                raw = "".join(s["text"] for s in spans)
                if not raw.strip():
                    continue
                if not (9.5 <= spans[0]["size"] <= 10.5):
                    continue
                if raw.strip().startswith("Under review as a conference"):
                    continue
                pages[pno].append(round(line["bbox"][1], 2))
    return pages


def measure(path):
    import fitz
    doc = fitz.open(str(path))
    lines = body_lines(doc)
    lefts = collections.Counter(round(b[0]) for b in lines)
    left = lefts.most_common(1)[0][0]

    # Only lines that do not end in a space say anything about the margin.
    measured = body_lines(doc, drop_trailing_space=True)
    right = max(b[2] for b in measured)

    deltas = []
    for ys in body_lines_by_page(doc).values():
        ys = sorted(ys)
        deltas += [round(b - a, 2) for a, b in zip(ys, ys[1:]) if 8 < b - a < 20]
    out = {
        "page_width": round(doc[0].rect.width / 72, 3),
        "page_height": round(doc[0].rect.height / 72, 3),
        "text_left": round(left / 72, 3),
        "text_right": round(right / 72, 3),
        "text_width": round((right - left) / 72, 3),
        "leading_pt": round(statistics.median(deltas), 1) if deltas else None,
        "pages": doc.page_count,
    }
    heads = sum(1 for p in doc
                if "Under review as a conference paper at ICLR 2027"
                in p.get_text("text", clip=__import__("fitz").Rect(
                    0, 0, p.rect.width, 90)))
    out["running_heads"] = heads
    fonts = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for s in line["spans"]:
                    if s["text"].strip():
                        fonts[s["font"]] += 1
    out["top_font"] = fonts.most_common(1)[0] if fonts else None
    doc.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-render", action="store_true",
                    help="measure an existing main_from_word.pdf")
    args = ap.parse_args(argv)

    if not args.skip_render and render() is None:
        return 1

    got = measure(RENDERED)
    print("\nwhat Word rendered")
    problems = []
    for key, want in EXPECTED.items():
        ok = abs(got[key] - want) <= TOL_IN
        if not ok:
            problems.append(key)
        print("  %-9s %-14s %8.3f in  (want %.3f)"
              % ("ok" if ok else "FAIL", key, got[key], want))

    ok = got["text_right"] <= RIGHT_MARGIN_IN + RIGHT_OVERFLOW_IN
    problems += [] if ok else ["right margin"]
    print("  %-9s %-14s %8.3f in  (must not pass %.3f; ragged-right may stop "
          "short)" % ("ok" if ok else "FAIL", "right margin", got["text_right"],
                      RIGHT_MARGIN_IN))

    ok = got["leading_pt"] is not None and abs(got["leading_pt"] - 11.0) <= 0.6
    problems += [] if ok else ["leading"]
    print("  %-9s %-14s %8s pt  (want 11.0)"
          % ("ok" if ok else "FAIL", "leading", got["leading_pt"]))

    ok = got["running_heads"] == got["pages"]
    problems += [] if ok else ["running head"]
    print("  %-9s %-14s %4d of %d pages"
          % ("ok" if ok else "FAIL", "running head", got["running_heads"],
             got["pages"]))

    font, n = got["top_font"]
    ok = "Times" in font or "TimesNewRoman" in font.replace(" ", "")
    problems += [] if ok else ["body font"]
    print("  %-9s %-14s %s (%d runs)"
          % ("ok" if ok else "FAIL", "body font", font, n))

    if LATEX_PDF.exists():
        import fitz
        with fitz.open(str(LATEX_PDF)) as tex:
            print("\n  Word %d pages vs LaTeX %d. Word breaks lines "
                  "differently; the\n  page limit is enforced on the LaTeX "
                  "PDF, which is the submission." % (got["pages"],
                                                     tex.page_count))

    print()
    if problems:
        print("%d failed: %s" % (len(problems), ", ".join(problems)))
        return 1
    print("Word renders the document on the venue's frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

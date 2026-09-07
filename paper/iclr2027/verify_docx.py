"""Check the .docx against the PDF it was converted from.

A conversion that silently drops a table, leaves a citation unresolved, or
embeds a figure Word cannot draw still opens cleanly, which is what makes it
worth checking rather than eyeballing.

    python verify_docx.py docx/main.docx

Three groups:

  * Page setup, against the dimensions measured off the PDF.
  * Content parity, against the PDF and the LaTeX source: section headings,
    tables, figures, and every number in Tables 1-3.
  * Breakage: unresolved citations or cross-references, images in a format
    Word will not render, and text the converter dropped.

Exit status is non-zero if anything fails.
"""

import pathlib
import re
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
PDF = HERE / "out" / "main.pdf"
TEX = HERE / "main.tex"

# Inches, measured off out/main.pdf. Word stores twips, 1440 per inch.
EXPECTED = {
    "page_width": 8.5,
    "page_height": 11.0,
    "left_margin": 1.5,
    "right_margin": 1.5,
    "top_margin": 1.171,
    "header_distance": 0.386,
}
TOL_IN = 0.01

RUNNING_HEAD = "Under review as a conference paper at ICLR 2027"
BODY_PT = 10.0
FONT = "Times New Roman"

# Word renders none of these; a figure in one is an empty box on the page.
UNRENDERABLE = (".pdf", ".eps", ".ps", ".svg")


class Report:
    def __init__(self):
        self.problems = []

    def check(self, name, ok, detail=""):
        print("  %-9s %-46s %s" % ("ok" if ok else "FAIL", name, detail))
        if not ok:
            self.problems.append(name)
        return ok


def page_setup(doc, rep):
    print("\npage setup")
    section = doc.sections[0]
    got = {
        "page_width": section.page_width.inches,
        "page_height": section.page_height.inches,
        "left_margin": section.left_margin.inches,
        "right_margin": section.right_margin.inches,
        "top_margin": section.top_margin.inches,
        "header_distance": section.header_distance.inches,
    }
    for key, want in EXPECTED.items():
        rep.check(key, abs(got[key] - want) <= TOL_IN,
                  "%.3f in (want %.3f)" % (got[key], want))
    measure = got["page_width"] - got["left_margin"] - got["right_margin"]
    rep.check("text measure", abs(measure - 5.5) <= TOL_IN,
              "%.3f in (the venue's 5.5)" % measure)

    head = "\n".join(p.text for p in section.header.paragraphs)
    rep.check("running head", RUNNING_HEAD in head, repr(head.strip()[:52]))

    normal = doc.styles["Normal"]
    rep.check("body font", normal.font.name == FONT, str(normal.font.name))
    rep.check("body size", normal.font.size.pt == BODY_PT,
              "%.1f pt" % normal.font.size.pt)


def document_text(path):
    """All visible text, including the parts inside math.

    python-docx exposes `w:t` runs only. Every number the converter set as an
    equation lands in `m:t` inside an `m:oMath`, which Word renders and
    python-docx reports as an empty string. Reading the XML directly is the
    difference between checking the document and checking a subset of it; the
    first pass here reported four table values as missing when all four were
    present as math.
    """
    import xml.etree.ElementTree as ET
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in ("word/document.xml", "word/header1.xml",
                     "word/footer1.xml"):
            if name not in z.namelist():
                continue
            root = ET.fromstring(z.read(name))
            for el in root.iter():
                if el.tag in (W, M) and el.text:
                    parts.append(el.text)
    text = "".join(parts)
    # Math sets a real minus (U+2212) and TeX ligatures reach the file as
    # single glyphs. Fold them to ASCII so a comparison against the source
    # tests the value rather than the typography.
    for uni, ascii_ in (("−", "-"), ("–", "-"), ("—", "-"),
                        ("‘", "'"), ("’", "'"),
                        ("“", '"'), ("”", '"'),
                        ("ﬁ", "fi"), ("ﬂ", "fl"),
                        (" ", " ")):
        text = text.replace(uni, ascii_)
    return text


def content_parity(doc, path, rep):
    print("\ncontent")
    text = document_text(path)

    tex = TEX.read_text(encoding="utf-8")
    want_sections = re.findall(r"\\section\*?\{([^}]*)\}", tex)
    missing = [s for s in want_sections
               if s.split("{")[0][:18].strip().lower() not in text.lower()]
    rep.check("every section heading present",
              not missing, "%d of %d" % (len(want_sections) - len(missing),
                                         len(want_sections))
              + ("; missing %s" % missing[:3] if missing else ""))

    want_tables = tex.count("\\begin{tabular}")
    rep.check("tables carried over", len(doc.tables) >= want_tables,
              "%d in docx, %d tabular in source" % (len(doc.tables),
                                                    want_tables))

    with zipfile.ZipFile(path) as z:
        media = [n for n in z.namelist() if n.startswith("word/media/")]
    want_figs = tex.count("\\includegraphics")
    rep.check("figures embedded", len(media) >= want_figs,
              "%d images, %d includegraphics" % (len(media), want_figs))
    bad = [m for m in media if m.lower().endswith(UNRENDERABLE)]
    rep.check("figures in a format Word renders", not bad,
              "all %d raster" % len(media) if not bad else str(bad[:3]))

    # Every number the tables report must survive the conversion.
    numbers = []
    for block in re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", tex,
                            re.S):
        numbers += re.findall(r"[+-]?\d+\.\d{3,6}", block)
    absent = [n for n in set(numbers) if n.lstrip("+") not in text]
    rep.check("table numbers survive conversion", not absent,
              "%d checked" % len(set(numbers))
              + ("; absent %s" % absent[:4] if absent else ""))
    return text


def breakage(doc, text, rep):
    print("\nbreakage")
    rep.check("no unresolved cross-references", "??" not in text,
              "none" if "??" not in text else "found '??'")
    unresolved = re.findall(r"\[CITATION[^\]]*\]|\\cite[a-z]*\{", text)
    rep.check("no unresolved citations", not unresolved,
              "none" if not unresolved else str(unresolved[:3]))
    rep.check("no raw LaTeX commands left", "\\begin{" not in text,
              "none" if "\\begin{" not in text else "found \\begin{")

    # A citation that citeproc resolved appears as an author-year string.
    cited = set(re.findall(r"\\citep?\{([^}]+)\}",
                           TEX.read_text(encoding="utf-8")))
    keys = {k.strip() for group in cited for k in group.split(",")}
    surnames = {"vaswani": "Vaswani", "michel2019sixteen": "Michel",
                "zhai2026xsa": "Zhai", "ainslie2023gqa": "Ainslie",
                "holm1979simple": "Holm"}
    absent = [v for k, v in surnames.items()
              if any(k in key for key in keys) and v not in text]
    rep.check("citations rendered", not absent,
              "%d keys cited" % len(keys)
              + ("; absent %s" % absent if absent else ""))

    # The three repairs the converter needs. Each was a visible defect in the
    # first build, so each is checked rather than assumed to have held.
    rep.check("no stray maketitle fragment",
              "maketitle" not in text and "aketitle" not in text,
              "none" if "aketitle" not in text else "found it")
    numbered = [p.text for p in doc.paragraphs
                if p.style.name == "Heading 4"
                and re.match(r"^\d+(\.\d+)*\s", p.text)]
    rep.check("run-in headings unnumbered", not numbered,
              "%d run-ins" % sum(1 for p in doc.paragraphs
                                 if p.style.name == "Heading 4")
              + ("; numbered %s" % numbered[:2] if numbered else ""))
    front = " ".join(p.text for p in doc.paragraphs[:6])
    rep.check("author block matches the PDF's anonymity notice",
              "Anonymous authors" in front
              and "double-blind review" in front
              and "anon@example" not in front,
              repr(front.replace("\n", " / ")[:70]))

    if PDF.exists():
        import fitz
        with fitz.open(str(PDF)) as pdf:
            pdf_words = len("".join(p.get_text() for p in pdf).split())
        ratio = len(text.split()) / pdf_words if pdf_words else 0
        # The PDF text includes the line-number ruler and repeated running
        # heads, which the docx holds once, so parity is a band not a point.
        rep.check("word count in range", 0.75 <= ratio <= 1.15,
                  "%d docx vs %d pdf, ratio %.2f"
                  % (len(text.split()), pdf_words, ratio))


def main(path):
    import docx
    path = pathlib.Path(path)
    doc = docx.Document(str(path))
    rep = Report()
    page_setup(doc, rep)
    text = content_parity(doc, path, rep)
    breakage(doc, text, rep)

    print()
    if rep.problems:
        print("%d checks failed: %s" % (len(rep.problems),
                                        ", ".join(rep.problems)))
        return 1
    print("the docx matches the PDF's page setup and carries its content")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else HERE / "docx" / "main.docx"))

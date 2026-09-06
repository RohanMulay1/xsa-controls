"""Measure page geometry and font sizes from a PDF, so the draft can be
compared against the venue template rather than eyeballed."""
import sys, collections, json
import fitz

def measure(path, pages=None):
    d = fitz.open(path)
    out = {"file": path, "n_pages": d.page_count, "pages": []}
    for pno, page in enumerate(d):
        if pages and pno not in pages:
            continue
        r = page.rect
        spans = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if s["text"].strip():
                        spans.append(s)
        xs = sorted({round(s["bbox"][0], 1) for s in spans})
        # column detection: cluster left edges
        cols = []
        for x in xs:
            if not cols or x - cols[-1][-1] > 40:
                cols.append([x])
            else:
                cols[-1].append(x)
        sizes = collections.Counter(round(s["size"], 1) for s in spans)
        out["pages"].append({
            "page": pno,
            "mediabox": [round(r.width, 1), round(r.height, 1)],
            "x_min": round(min(s["bbox"][0] for s in spans), 1),
            "x_max": round(max(s["bbox"][2] for s in spans), 1),
            "y_min": round(min(s["bbox"][1] for s in spans), 1),
            "y_max": round(max(s["bbox"][3] for s in spans), 1),
            "col_left_edges": [round(c[0], 1) for c in cols],
            "sizes": sizes.most_common(10),
            "fonts": collections.Counter(s["font"] for s in spans).most_common(6),
        })
    return out

if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[1]), indent=1))

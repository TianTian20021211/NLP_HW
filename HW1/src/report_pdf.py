"""Render ``report.md`` to ``report.pdf`` (PDF deliverable per PDF §6).

Uses ``markdown2`` for GFM-table parsing and ``weasyprint`` for HTML→PDF.
Resolves relative image paths (``data/cache/plots/*.png``) against the
HW1 root so the PDF embeds the actual figures from the cache.
"""

from __future__ import annotations

from pathlib import Path

import markdown2
from weasyprint import HTML

from .io_paths import HW1_ROOT, error


REPORT_MD = HW1_ROOT / "report.md"
REPORT_PDF = HW1_ROOT / "report.pdf"


_CSS = """
@page { size: A4; margin: 18mm 18mm 20mm 18mm; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
       font-size: 10pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 22pt; margin-top: 0; }
h2 { font-size: 15pt; margin-top: 18pt; border-bottom: 1px solid #ccc;
     padding-bottom: 2pt; }
h3 { font-size: 12pt; margin-top: 12pt; }
h4 { font-size: 11pt; margin-top: 9pt; }
p, li { font-size: 10pt; }
code { font-family: 'DejaVu Sans Mono', 'Liberation Mono', monospace;
       font-size: 9pt; background: #f4f4f4; padding: 1px 3px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 8pt; border-radius: 4pt;
      font-size: 8.5pt; overflow-x: auto; }
pre code { background: transparent; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0;
        font-size: 9pt; }
th, td { border: 1px solid #c0c0c0; padding: 4pt 6pt; text-align: left; }
th { background: #ececec; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; height: auto; display: block;
      margin: 8pt auto; page-break-inside: avoid; }
strong { color: #000; }
a { color: #1a4f8b; text-decoration: none; }
hr { border: none; border-top: 1px solid #d0d0d0; margin: 12pt 0; }
.figure-caption { font-size: 8.5pt; color: #555; text-align: center;
                  margin-top: -4pt; margin-bottom: 8pt; }
"""


def render(md_path: Path = REPORT_MD, pdf_path: Path = REPORT_PDF) -> Path:
    """Convert ``md_path`` to ``pdf_path``. Returns the output path.

    Tables, fenced code blocks, and inline images are passed through
    explicit markdown2 extras so GFM-style markdown renders correctly.
    The HTML is wrapped in a ``<style>`` block from ``_CSS`` and given a
    ``base_url`` of the markdown's parent directory so relative image
    paths (e.g. ``data/cache/plots/equity_curves.png``) resolve cleanly.
    """
    if not md_path.is_file():
        error(f"report markdown missing: {md_path}")
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown2.markdown(
        md_text,
        extras=["tables", "fenced-code-blocks", "header-ids", "strike",
                "cuddled-lists", "code-friendly", "break-on-newline"],
    )
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>HW1 Report</title>"
        f"<style>{_CSS}</style></head><body>"
        f"{html_body}"
        "</body></html>"
    )
    HTML(string=html_doc, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
    return pdf_path


if __name__ == "__main__":
    out = render()
    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KiB)")

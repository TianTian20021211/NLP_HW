"""Phase 7 PDF generation via markdown -> HTML -> weasyprint.

Converts docs/report.md to a PDF using the markdown library for markdown-to-HTML
and weasyprint for HTML-to-PDF rendering. Charts embedded in the markdown as
relative ![](figures/xxx.png) references are resolved automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from data.config import AUDIT_DIR, FIGURES_DIR, PROJECT_ROOT, REPORT_PDF, REPORTS_DIR, RESULTS_DIR

DOCS_DIR = PROJECT_ROOT / "docs"


def _check_weasyprint() -> None:
    try:
        import weasyprint  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "weasyprint is required for PDF generation.\n"
            "  pip install weasyprint\n"
            "On Ubuntu/Debian you may also need system dependencies:\n"
            "  sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev libcairo2"
        ) from None


def build_html_body(md_path: Path) -> str:
    text = md_path.read_text()
    import markdown
    return markdown.markdown(text, extensions=["tables", "fenced_code", "codehilite"])


def wrap_html_document(
    html_body: str,
    title: str = "ProntoNLP ATC Signal Backtest Report",
    checklist_html: str | None = None,
) -> str:
    checklist_section = ""
    if checklist_html:
        checklist_section = (
            '<section style="page-break-before: always;">'
            "<h2>Appendix: Audit Checklist</h2>"
            f"{checklist_html}"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{
    size: A4;
    margin: 1.8cm 1.5cm;
    @bottom-center {{
      content: "Page " counter(page);
      font-size: 7.5pt;
      font-family: 'DejaVu Sans', sans-serif;
      color: #888;
    }}
  }}
  @page :first {{
    @bottom-center {{
      content: none;
    }}
  }}
  body {{
    font-family: 'DejaVu Serif', serif;
    font-size: 9pt;
    line-height: 1.35;
    color: #222;
  }}
  h1, h2, h3, h4 {{
    font-family: 'DejaVu Sans', sans-serif;
    color: #111;
  }}
  h1 {{ font-size: 15pt; margin-top: 0; }}
  h2 {{ font-size: 12pt; margin-top: 1.2em; }}
  h3 {{ font-size: 10pt; }}
  h1 + h2 {{ page-break-before: avoid; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 7.5pt;
    margin: 6px 0;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #999;
    padding: 2px 4px;
    text-align: left;
    vertical-align: top;
  }}
  th {{ background: #f0f0f0; font-weight: 600; }}
  img {{
    max-width: 100%;
    margin: 8px 0;
    page-break-inside: avoid;
  }}
  code {{
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 7.5pt;
    background: #f4f4f4;
    padding: 1px 3px;
    border-radius: 2px;
  }}
  pre {{
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 7.5pt;
    background: #f4f4f4;
    padding: 6px 10px;
    border: 1px solid #ddd;
    overflow-x: auto;
  }}
  p {{ text-align: justify; }}
  table.img-pair {{ border: none; width: 100%; margin: 4px 0; page-break-inside: avoid; }}
  table.img-pair td {{ border: none; width: 50%; text-align: center; vertical-align: top; padding: 2px 4px; }}
  table.img-pair td strong {{ font-size: 7.5pt; display: block; margin-bottom: 2px; }}
  table.img-pair img {{ max-width: 100%; height: auto; margin: 0; }}
</style>
</head>
<body>
{html_body}
{checklist_section}
</body>
</html>"""


def _check_figure_files(md_path: Path) -> None:
    """Scan markdown for ![](...) references and warn about missing figure files."""
    text = md_path.read_text()
    refs = re.findall(r"!\[.*?\]\(([^)]+)\)", text)
    if not refs:
        return
    print(f"  Checking {len(refs)} figure reference(s)...")
    for ref in refs:
        # Resolve relative paths against the markdown's directory
        candidate = (md_path.parent / ref).resolve()
        if not candidate.exists():
            print(f"  WARNING: referenced figure not found: {candidate}")


def generate_pdf(
    report_md: Path | None = None,
    output_pdf: Path | None = None,
    checklist_md: Path | None = None,
) -> Path:
    _check_weasyprint()
    from weasyprint import HTML

    report_md = report_md or (DOCS_DIR / "report.md")
    output_pdf = output_pdf or REPORT_PDF

    if not report_md.exists():
        raise FileNotFoundError(f"Report markdown not found: {report_md}")

    print(f"  Reading: {report_md}")
    body = build_html_body(report_md)

    _check_figure_files(report_md)

    checklist_html = None
    # Checklist is now inline in report.md — skip external append to avoid duplication.
    # To restore external append, pass --checklist-md explicitly.
    checklist_path = checklist_md  # only use explicit path, no default
    if checklist_path is not None and checklist_path.exists():
        print(f"  Appending checklist: {checklist_path}")
        checklist_html = build_html_body(checklist_path)

    title = "ProntoNLP ATC Signal Backtest Report"
    print(f"  Building HTML document...")
    html = wrap_html_document(body, title=title, checklist_html=checklist_html)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Rendering PDF -> {output_pdf}")
    HTML(string=html, base_url=str(REPORTS_DIR)).write_pdf(output_pdf)
    print(f"  Done: {output_pdf} ({output_pdf.stat().st_size:,} bytes)")
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDF report from markdown")
    parser.add_argument("--report-md", type=Path, default=DOCS_DIR / "report.md")
    parser.add_argument("--output", type=Path, default=REPORT_PDF)
    parser.add_argument("--checklist-md", type=Path, default=None, help="Optional external checklist .md to append (inline in report.md by default)")
    args = parser.parse_args()

    try:
        generate_pdf(
            report_md=args.report_md,
            output_pdf=args.output,
            checklist_md=args.checklist_md,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

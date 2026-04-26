"""
Convert every PDF in ./books/ to Markdown in ./books_text/

Usage:
    python convert_to_markdown.py
"""

import sys
from pathlib import Path

from markitdown import MarkItDown

SRC_DIR = Path("books")
OUT_DIR = Path("books_text")


def convert_one(md: MarkItDown, pdf_path: Path, out_path: Path) -> int:
    """Convert a single PDF and return the number of lines written."""
    result = md.convert(str(pdf_path))
    text = result.text_content
    out_path.write_text(text, encoding="utf-8")
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR}/ not found. Run download_books.py first.",
              file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    md = MarkItDown()

    pdfs = sorted(SRC_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs to convert.\n")

    for i, pdf in enumerate(pdfs, 1):
        out = OUT_DIR / f"{pdf.stem}.md"

        if out.exists():
            print(f"[{i}/{len(pdfs)}] SKIP  {pdf.name}  (already converted)")
            continue

        print(f"[{i}/{len(pdfs)}] CONV  {pdf.name}")
        try:
            n_lines = convert_one(md, pdf, out)
            print(f"            wrote -> {out}  ({n_lines:,} lines)")
        except Exception as e:
            print(f"            FAILED: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
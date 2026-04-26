"""
Chunk the markdown books in ./books_text/ for RAG.

For each .md file:
  1. Read the file
  2. Split into lines
  3. Drop empty / whitespace-only lines
  4. Build a {"source": <filename>, "content": [lines]} dict

Then run gitsource.chunk_documents(size=100, step=50) on the list and
report how many chunks each book produced.

Usage:
    python chunk_books.py
"""

import sys
from pathlib import Path

from gitsource import chunk_documents

SRC_DIR = Path("books_text")
SIZE = 100
STEP = 50


def load_book(md_path: Path) -> dict:
    """Read a markdown file and return {'source': ..., 'content': [non-empty lines]}."""
    raw = md_path.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if line.strip()]
    return {"source": md_path.name, "content": lines}


def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR}/ not found. Run convert_to_markdown.py first.",
              file=sys.stderr)
        return 1

    md_files = sorted(SRC_DIR.glob("*.md"))
    if not md_files:
        print(f"ERROR: no .md files found in {SRC_DIR}/", file=sys.stderr)
        return 1

    documents = [load_book(p) for p in md_files]

    # Per-book report
    print(f"{'book':35s}  {'lines':>8s}  {'chunks':>7s}")
    print("-" * 55)
    for doc in documents:
        single = chunk_documents([doc], size=SIZE, step=STEP)
        print(f"{doc['source']:35s}  {len(doc['content']):>8,}  {len(single):>7,}")

    # All books together
    all_chunks = chunk_documents(documents, size=SIZE, step=STEP)
    print("-" * 55)
    print(f"TOTAL chunks across all books: {len(all_chunks):,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
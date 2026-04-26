"""
Index the chunked books with minsearch.

Pipeline (continues from chunk_books.py):
  1. Read every .md in books_text/
  2. Drop empty/whitespace lines
  3. Build {"source", "content": [lines]} per book
  4. Chunk with gitsource (size=100, step=50)
  5. Stringify each chunk's content (list -> "\n"-joined string)
  6. Build a minsearch.Index and fit it

Usage:
    uv run python index_chunks.py
"""

import sys
from pathlib import Path

from gitsource import chunk_documents
from minsearch import Index

SRC_DIR = Path("books_text")
SIZE = 100
STEP = 50


def load_book(md_path: Path) -> dict:
    """{'source': filename, 'content': [non-empty lines]}"""
    lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"source": md_path.name, "content": lines}


def prepare_documents(chunks: list[dict]) -> list[dict]:
    """Turn each chunk's `content` (list of lines) into a single string."""
    docs = []
    for c in chunks:
        docs.append({
            "source": c["source"],
            "start":  c["start"],
            "content": "\n".join(c["content"]),
        })
    return docs


def main() -> int:
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR}/ not found.", file=sys.stderr)
        return 1

    md_files = sorted(SRC_DIR.glob("*.md"))
    documents = [load_book(p) for p in md_files]
    chunks = chunk_documents(documents, size=SIZE, step=STEP)
    print(f"Produced {len(chunks):,} chunks from {len(documents)} books.")

    docs = prepare_documents(chunks)

    # `content` is the searchable text; `source` is a categorical filter
    index = Index(
        text_fields=["content"],
        keyword_fields=["source"],
    )
    index.fit(docs)

    print(f"Indexed {len(docs):,} documents.")

    # Sanity-check the index actually works
    results = index.search("recursion", num_results=3)
    print(f"\nSample query 'recursion' -> {len(results)} hits:")
    for r in results:
        preview = r["content"].replace("\n", " ")[:80]
        print(f"  [{r['source']}] {preview}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
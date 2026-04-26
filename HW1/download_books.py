"""
Download PDFs listed in books.csv from the AI Engineering Buildcamp homework.

Reads a CSV with columns: title, book_url, pdf_url
Downloads each pdf_url to ./books/<safe-title>.pdf
"""

import csv
import re
import sys
from pathlib import Path

import requests

CSV_PATH = Path("books.csv")
OUT_DIR = Path("books")
TIMEOUT = 30          # seconds per request
CHUNK_SIZE = 1 << 15  # 32 KB streaming chunks


def slugify(name: str) -> str:
    """Turn a book title into a safe lowercase filename."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def download(url: str, dest: Path) -> None:
    """Stream-download a single file to dest."""
    headers = {"User-Agent": "ai-buildcamp-homework/1.0"}
    with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers) as resp:
        resp.raise_for_status()
        with dest.open("wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run wget for books.csv first.",
              file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)

    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Found {len(rows)} books to download.\n")

    failures = []
    for i, row in enumerate(rows, 1):
        title = row["title"]
        url = row["pdf_url"]
        dest = OUT_DIR / f"{slugify(title)}.pdf"

        if dest.exists():
            print(f"[{i}/{len(rows)}] SKIP  {title}  (already at {dest})")
            continue

        print(f"[{i}/{len(rows)}] GET   {title}")
        try:
            download(url, dest)
            size_mb = dest.stat().st_size / 1_000_000
            print(f"            saved -> {dest}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"            FAILED: {e}")
            failures.append((title, url, str(e)))
            if dest.exists():
                dest.unlink()  # don't leave a partial file behind

    ok = len(rows) - len(failures)
    print(f"\nDone. {ok}/{len(rows)} downloaded.")
    if failures:
        print("\nFailures:")
        for title, url, err in failures:
            print(f"  - {title} ({url}): {err}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
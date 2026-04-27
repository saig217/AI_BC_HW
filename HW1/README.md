# AI Engineering Buildcamp — Homework 1

A small RAG (Retrieval-Augmented Generation) system built over Allen Downey's
free *Think* book series. Downloads the PDFs, converts them to markdown,
chunks them, indexes the chunks with TF-IDF, and answers questions using
OpenAI's `gpt-4o-mini` — both as plain text and as a structured Pydantic
response.

## What it does

```
PDFs ──> Markdown ──> Lines ──> Chunks ──> Index ──> Search ──> RAG ──> Answer
```

Each step is a standalone script you can run in order. Outputs of one step
feed into the next.

## Stack

- **`uv`** — project & dependency manager
- **`requests`** — PDF downloads
- **`markitdown[pdf]`** — PDF → Markdown conversion
- **`gitsource`** — sliding-window chunking
- **`minsearch`** — lightweight TF-IDF index
- **`openai`** — LLM calls via the Responses API
- **`pydantic`** — structured-output schema
- **`python-dotenv`** — `.env` loading

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <your-repo-name>

# 2. Install dependencies
uv sync
# or, if starting fresh:
uv add requests 'markitdown[pdf]' gitsource minsearch openai pydantic python-dotenv

# 3. Add your OpenAI API key
echo 'OPENAI_API_KEY=sk-your-key-here' > .env
```

## Usage

Run the scripts in order. Each is idempotent — safe to re-run if a step fails
partway.

```bash
# 1. Download the seven PDFs to ./books/
wget https://raw.githubusercontent.com/alexeygrigorev/ai-engineering-buildcamp-code/main/01-foundation/homework/books.csv
uv run python download_books.py

# 2. Convert PDFs -> Markdown in ./books_text/
uv run python convert_to_markdown.py

# 3. Chunk each book and report counts
uv run python chunk_books.py

# 4. Build the search index and run a sanity-check query
uv run python index_chunks.py

# 5. Run a search and show the top hit
uv run python search_index.py

# 6. Full RAG with token usage tracking
uv run python rag_query.py

# 7. RAG with Pydantic-structured output and token comparison
uv run python rag_structured.py
```

## Project layout

```
.
├── README.md
├── pyproject.toml
├── .env                    # not committed; holds OPENAI_API_KEY
├── books.csv               # input: titles + PDF URLs
├── books/                  # generated: downloaded PDFs
├── books_text/             # generated: markdown conversions
├── download_books.py
├── convert_to_markdown.py
├── chunk_books.py
├── index_chunks.py
├── search_index.py
├── rag_query.py
└── rag_structured.py
```

## Results

| Q | Question                                              | Answer  |
|---|-------------------------------------------------------|---------|
| 1 | Lines in `think_python_2e.md`                         | 14,268  |
| 2 | Chunks for Think Python (size=100, step=50)           | 214     |
| 3 | Total chunks indexed across all books                 | ~1,000  |
| 4 | Top result for `"python function definition"`         | Think Python |
| 5 | Input tokens for one RAG query                        | ~7,300  |
| 6 | Extra input tokens for structured output              | ~190    |

Numbers may differ slightly from one machine to another due to `pdfminer.six`
version differences in PDF text extraction.

## Design notes

**Chunk math (`gitsource.chunk_documents`).** The library uses a sliding
window that yields one chunk per `step`, plus one final tail chunk. For
`size=100, step=50`:

```
chunks = (n_lines − 100) // 50 + 2     for n_lines ≥ 100
```

**Why join lines into one string before indexing.** `gitsource` works on
*lists* of items so the sliding window can count discrete units; `minsearch`
works on *strings* so it can tokenize them for TF-IDF. The join (`"\n".join`)
is just a format conversion at the boundary — it doesn't undo the chunking.

**Cost of structured outputs.** Sending a Pydantic schema with the request
adds ~190 input tokens for this 6-field model. Output tokens also grow
because the response now includes JSON keys, a confidence score, an answer
category, and follow-up questions in addition to the prose answer. In
exchange, the response is guaranteed to parse as valid JSON matching the
schema — no defensive `try/except` around `json.loads` in production code.

**Retrieval quality.** Plain TF-IDF over PDF-extracted markdown picks up a
non-trivial amount of noise — index pages, tables of contents, code listings
— competing with real prose for top-K slots. A real production system would
add chunk-quality filtering and/or hybrid retrieval (TF-IDF + dense
embeddings) and/or reranking with a cross-encoder.

## Books used

All books are by [Allen Downey](https://greenteapress.com/) and freely
distributed under permissive licenses:

- Think Python 2e
- Think DSP
- Think Complexity 2e
- Think Java 2e
- Physical Modeling in MATLAB
- Think OS
- Think C++

## License

Code in this repo is MIT-licensed. The books themselves are owned by
their respective authors and used here solely for educational purposes.
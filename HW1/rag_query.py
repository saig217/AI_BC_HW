"""
Question 5 — Full RAG with token usage tracking.

Pipeline:
  load books -> chunk -> stringify -> index -> search -> build prompt -> call LLM
  + report input/output token counts from the OpenAI response

Requires the OPENAI_API_KEY environment variable to be set.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python rag_query.py
"""

import json
import os
import sys
from pathlib import Path

from gitsource import chunk_documents
from minsearch import Index
from openai import OpenAI

# Load .env if present (no-op if python-dotenv isn't installed)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- pipeline config ---------------------------------------------------------
SRC_DIR = Path("books_text")
SIZE = 100
STEP = 50
QUERY = "python function definition"
NUM_RESULTS = 5
MODEL = "gpt-4o-mini"

# --- prompts (verbatim from the homework) ------------------------------------
instructions = """
You're a course assistant, your task is to answer the QUESTION from the
course students using the provided CONTEXT
"""

prompt_template = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()


# --- pipeline pieces ---------------------------------------------------------
def load_book(md_path: Path) -> dict:
    lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"source": md_path.name, "content": lines}


def prepare_documents(chunks: list[dict]) -> list[dict]:
    return [
        {"source": c["source"], "start": c["start"],
         "content": "\n".join(c["content"])}
        for c in chunks
    ]


def build_index() -> Index:
    md_files = sorted(SRC_DIR.glob("*.md"))
    documents = [load_book(p) for p in md_files]
    chunks = chunk_documents(documents, size=SIZE, step=STEP)
    docs = prepare_documents(chunks)
    idx = Index(text_fields=["content"], keyword_fields=["source"])
    idx.fit(docs)
    return idx


# --- RAG functions (homework code, lightly modified to return token counts) --
def build_prompt(question: str, search_results: list[dict]) -> str:
    context = json.dumps(search_results, indent=2)
    return prompt_template.format(question=question, context=context).strip()


def search(index: Index, question: str) -> list[dict]:
    return index.search(question, num_results=NUM_RESULTS)


def llm(client: OpenAI, user_prompt: str, instructions: str,
        model: str = MODEL) -> tuple[str, int, int]:
    """Call the model and return (text, input_tokens, output_tokens)."""
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.create(model=model, input=messages)
    return (
        response.output_text,
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


def rag(client: OpenAI, index: Index, query: str) -> tuple[str, int, int, str]:
    """Return (answer, input_tokens, output_tokens, prompt_for_inspection)."""
    search_results = search(index, query)
    prompt = build_prompt(query, search_results)
    answer, in_tok, out_tok = llm(client, prompt, instructions)
    return answer, in_tok, out_tok, prompt


# --- main --------------------------------------------------------------------
def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        print("", file=sys.stderr)
        print("Try one of these:", file=sys.stderr)
        print("  1. Same-shell export:  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        print("                         uv run python rag_query.py", file=sys.stderr)
        print("  2. .env file:          uv add python-dotenv", file=sys.stderr)
        print("                         echo 'OPENAI_API_KEY=sk-...' > .env", file=sys.stderr)
        print("                         uv run python rag_query.py", file=sys.stderr)
        print("  3. Inline one-shot:    OPENAI_API_KEY=sk-... uv run python rag_query.py",
              file=sys.stderr)
        return 1
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR}/ not found.", file=sys.stderr)
        return 1

    print("Building index...")
    index = build_index()
    client = OpenAI()

    print(f"Running RAG for: {QUERY!r}\n")
    answer, in_tok, out_tok, prompt = rag(client, index, QUERY)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)
    print()
    print("=" * 70)
    print("TOKEN USAGE")
    print("=" * 70)
    print(f"  input  tokens: {in_tok:>6,}")
    print(f"  output tokens: {out_tok:>6,}")
    print(f"  total  tokens: {in_tok + out_tok:>6,}")
    print()
    print(f"(prompt char length, for sanity: {len(prompt):,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
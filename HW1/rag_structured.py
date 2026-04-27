"""
Question 6 — Structured RAG output via Pydantic + token comparison.

Runs the same RAG query as Q5 twice — once unstructured, once with a Pydantic
schema — and prints the difference in input tokens.

Usage:
    uv run python rag_structured.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Literal

from gitsource import chunk_documents
from minsearch import Index
from openai import OpenAI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- config ------------------------------------------------------------------
SRC_DIR = Path("books_text")
SIZE = 100
STEP = 50
QUERY = "python function definition"
NUM_RESULTS = 5
MODEL = "gpt-4o-mini"

# --- prompts (same as Q5) ----------------------------------------------------
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


# --- response schema (verbatim from the homework) ----------------------------
class RAGResponse(BaseModel):
    answer: str = Field(description="The main answer to the user's question in markdown")
    found_answer: bool = Field(description="True if relevant information was found in the documentation")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    confidence_explanation: str = Field(description="Explanation about the confidence level")
    answer_type: Literal["how-to", "explanation", "troubleshooting", "comparison", "reference"] = Field(
        description="The category of the answer"
    )
    followup_questions: list[str] = Field(description="Suggested follow-up questions")


# --- pipeline pieces ---------------------------------------------------------
def load_book(md_path: Path) -> dict:
    lines = [l for l in md_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {"source": md_path.name, "content": lines}


def prepare_documents(chunks):
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


def build_prompt(question: str, search_results) -> str:
    context = json.dumps(search_results, indent=2)
    return prompt_template.format(question=question, context=context).strip()


# --- two flavors of llm() ----------------------------------------------------
def llm_unstructured(client, user_prompt, instructions, model=MODEL):
    """Plain text response. Returns (text, in_tok, out_tok)."""
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.create(model=model, input=messages)
    return response.output_text, response.usage.input_tokens, response.usage.output_tokens


def llm_structured(client, user_prompt, instructions, model=MODEL):
    """Pydantic-validated response. Returns (RAGResponse, in_tok, out_tok)."""
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=RAGResponse,
    )
    return response.output_parsed, response.usage.input_tokens, response.usage.output_tokens


def rag(client, index, query, structured: bool):
    search_results = index.search(query, num_results=NUM_RESULTS)
    prompt = build_prompt(query, search_results)
    fn = llm_structured if structured else llm_unstructured
    answer, in_tok, out_tok = fn(client, prompt, instructions)
    return answer, in_tok, out_tok


# --- main --------------------------------------------------------------------
def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        return 1
    if not SRC_DIR.exists():
        print(f"ERROR: {SRC_DIR}/ not found.", file=sys.stderr)
        return 1

    print("Building index...")
    index = build_index()
    client = OpenAI()

    # Q5-style call
    print(f"\n[1/2] Unstructured RAG for: {QUERY!r}")
    _, in_unstruct, out_unstruct = rag(client, index, QUERY, structured=False)
    print(f"      input={in_unstruct:,}  output={out_unstruct:,}")

    # Q6-style call
    print(f"\n[2/2] Structured RAG for: {QUERY!r}")
    parsed, in_struct, out_struct = rag(client, index, QUERY, structured=True)
    print(f"      input={in_struct:,}  output={out_struct:,}")

    # Show the structured answer
    print("\n" + "=" * 70)
    print("STRUCTURED ANSWER (Pydantic-validated)")
    print("=" * 70)
    print(parsed.model_dump_json(indent=2))

    # The headline number for the homework
    print("\n" + "=" * 70)
    print("INPUT TOKEN COMPARISON")
    print("=" * 70)
    print(f"  Unstructured input tokens : {in_unstruct:>6,}")
    print(f"  Structured   input tokens : {in_struct:>6,}")
    print(f"  Difference (struct - un)  : {in_struct - in_unstruct:>+6,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
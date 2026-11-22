"""Prompts. The decision head was fitted on features produced with THESE exact strings,
so changing them invalidates the calibrator (tests/test_prompts.py guards against drift)."""

from __future__ import annotations

from collections.abc import Sequence

FORMAT = "Decision: <yes|no|maybe>\nSources: <doc_id>, <doc_id>"

SYSTEM_RAG = (
    "You are a biomedical literature assistant. Answer the research question using ONLY the provided abstracts.\n"
    "Decide: yes, no, or maybe. Use 'maybe' only when the abstracts report mixed or inconclusive evidence.\n"
    "In Sources list the doc_id of every abstract you relied on.\n"
    "Reply in exactly this format and nothing else:\n" + FORMAT + "\n"
    "This tool summarises published research. It does not give medical advice."
)

SYSTEM_EXPLAIN = (
    "You are a biomedical literature assistant. Using ONLY the provided abstracts, summarise in one or two "
    "sentences what they report about the research question. Do not give medical advice and do not go beyond "
    "what the abstracts say."
)

DISCLAIMER = (
    "Research summary generated from a fixed set of PubMed abstracts. Not medical advice; "
    "it can be wrong. Check the cited abstracts."
)


def format_context(hits: Sequence) -> str:
    """Group chunks by document and label each with its PubMed ID (same format as the evaluation notebooks)."""
    by_doc: dict[str, list[str]] = {}
    for hit in hits:
        by_doc.setdefault(hit.doc_id, []).append(hit.text)
    return "\n\n".join(f"[doc_id: {doc_id}]\n" + "\n...\n".join(texts) for doc_id, texts in by_doc.items())


def build_user_prompt(question: str, hits: Sequence) -> str:
    return f"ABSTRACTS:\n{format_context(hits)}\n\nQUESTION: {question}"

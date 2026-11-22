"""The calibrator was fitted on features produced with the evaluation notebook's prompts.
If these drift apart the decision head silently stops being valid, so compare them directly."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pubmed_rag import prompts

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "01_baseline_rag_eval.ipynb"


def _notebook_cell(marker: str) -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    hits = ["".join(c["source"]) for c in cells if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1
    return hits[0]


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="evaluation notebook not present")
def test_system_prompt_matches_the_notebook():
    src = _notebook_cell("SYSTEM_RAG = (")
    block = src[src.index("FORMAT ="): src.index("def format_context")]
    ns = {"WITH_REASON": False}
    exec(block, ns)  # noqa: S102 - trusted file from this repo
    assert prompts.SYSTEM_RAG == ns["SYSTEM_RAG"]


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="evaluation notebook not present")
def test_context_format_matches_the_notebook():
    src = _notebook_cell("def format_context")
    ns: dict = {}
    exec(src[src.index("def format_context"): src.index("def parse_answer")], ns)  # noqa: S102
    raw = [{"doc_id": "11", "text": "first abstract"}, {"doc_id": "22", "text": "second abstract"}]
    hits = [SimpleNamespace(**h) for h in raw]
    assert prompts.format_context(hits) == ns["format_context"](raw)
    assert prompts.build_user_prompt("Is X good?", hits) == f"ABSTRACTS:\n{ns['format_context'](raw)}\n\nQUESTION: Is X good?"


def test_prompt_fixes_the_answer_format():
    assert "Decision: <yes|no|maybe>" in prompts.SYSTEM_RAG and "ONLY the provided abstracts" in prompts.SYSTEM_RAG

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("requests")
DEMO = Path(__file__).resolve().parents[1] / "app" / "demo.py"
spec = importlib.util.spec_from_file_location("demo", DEMO)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)

RESULT = {
    "request_id": "abc", "decision": "no", "probabilities": {"yes": 0.2, "no": 0.7, "maybe": 0.1}, "calibrated": True,
    "low_confidence": False, "evidence": [{"rank": 1, "doc_id": "123", "score": 0.9, "text": "x" * 800}],
    "explanation": None, "warning": None, "disclaimer": "Not medical advice.", "backend": "local",
    "retrieval_mode": "dense", "corpus_size": 5000, "timings_ms": {"total": 900.0, "retrieval": 40.0, "scoring": 850.0, "explain": 0.0},
}


def test_format_result_links_to_pubmed_and_truncates():
    verdict, probs, evidence, details = demo.format_result(RESULT)
    assert "NO" in verdict and probs["no"] == 0.7
    assert "https://pubmed.ncbi.nlm.nih.gov/123/" in evidence and evidence.endswith("...")
    assert "5,000" in details and "calibrated" in details


def test_format_result_retrieval_only_and_warning():
    res = {**RESULT, "decision": None, "probabilities": None, "warning": "see a clinician", "evidence": []}
    verdict, probs, evidence, _ = demo.format_result(res)
    assert "No verdict" in verdict and "see a clinician" in verdict and probs is None and "nothing retrieved" in evidence


def test_demo_builds_when_gradio_is_installed():
    pytest.importorskip("gradio")
    assert demo.build_demo("http://127.0.0.1:1") is not None

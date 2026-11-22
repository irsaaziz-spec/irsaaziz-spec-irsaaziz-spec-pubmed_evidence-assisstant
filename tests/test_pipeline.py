import pytest
from conftest import FakeBackend, HashEmbedder

from pubmed_rag.config import Settings
from pubmed_rag.llm import NullBackend
from pubmed_rag.pipeline import NO_EVIDENCE_WARNING, Assistant
from pubmed_rag.retrieval import Retriever

Q = "Does topic5 treatment improve outcome?"


def test_answer_has_decision_probabilities_and_evidence(assistant):
    res = assistant.ask(Q)
    assert res["decision"] == "yes"  # logprobs (-0.5, -1.5, -3.0) favour yes
    assert sum(res["probabilities"].values()) == pytest.approx(1.0)
    assert res["calibrated"] is True and res["backend"] == "fake"
    assert len(res["evidence"]) == 5 and res["evidence"][0]["doc_id"] == "1005"
    assert set(res["timings_ms"]) == {"retrieval", "scoring", "explain", "total"}
    assert res["corpus_size"] == 80 and "medical advice" in res["disclaimer"].lower()


def test_prompt_contains_ranked_doc_ids_and_the_question(assistant):
    assistant.ask(Q)
    system, user = assistant.backend.score_calls[0]
    assert "does not give medical advice" in system
    assert user.startswith("ABSTRACTS:\n[doc_id: 1005]") and user.endswith(f"QUESTION: {Q}")


def test_retrieval_only_backend(retriever, settings):
    res = Assistant(retriever, NullBackend(), None, settings).ask(Q)
    assert res["decision"] is None and res["probabilities"] is None and len(res["evidence"]) == 5


def test_uncalibrated_fallback_flag(retriever, settings):
    res = Assistant(retriever, FakeBackend(), None, settings).ask(Q)
    assert res["calibrated"] is False and res["decision"] == "yes"


def test_low_confidence_flag(retriever, identity_calibrator):
    flat = FakeBackend(logprobs=(-1.0, -1.1, -1.2))  # nearly uniform
    res = Assistant(retriever, flat, identity_calibrator, Settings(low_confidence_threshold=0.5)).ask(Q)
    assert res["low_confidence"] is True


def test_personal_advice_warning(assistant):
    assert assistant.ask("Should I take aspirin for my headache?")["warning"]
    assert assistant.ask(Q)["warning"] is None


def test_no_evidence_skips_the_llm(retriever, identity_calibrator, settings):
    backend = FakeBackend()
    res = Assistant(retriever, backend, identity_calibrator, Settings(retrieval_mode="bm25")).ask("zzzz qqqq")
    assert res["evidence"] == [] and res["decision"] is None and res["warning"] == NO_EVIDENCE_WARNING
    assert backend.score_calls == []


def test_explain_calls_generate_only_when_asked(assistant):
    assert assistant.ask(Q)["explanation"] is None and assistant.backend.generate_calls == 0
    res = assistant.ask(Q, explain=True)
    assert res["explanation"] and assistant.backend.generate_calls == 1


def test_dense_falls_back_to_bm25_without_an_embedder(corpus, settings):
    a = Assistant(Retriever(corpus, embedder=None), NullBackend(), None, settings)
    assert a.mode == "bm25" and a.ask(Q)["retrieval_mode"] == "bm25"


def test_k_override(assistant):
    assert len(assistant.ask(Q, k=3)["evidence"]) == 3


def test_settings_validation():
    with pytest.raises(ValueError):
        Settings(backend="gpt")
    with pytest.raises(ValueError):
        Settings(retrieval_mode="magic")
    with pytest.raises(ValueError):
        Settings(k=0)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("PMRAG_BACKEND", "local")
    monkeypatch.setenv("PMRAG_K", "3")
    monkeypatch.setenv("PMRAG_LOAD_4BIT", "true")
    s = Settings.from_env()
    assert (s.backend, s.k, s.load_4bit) == ("local", 3, True)


def test_hash_embedder_is_deterministic():
    e = HashEmbedder()
    assert (e.encode_query("topic1") == e.encode_query("topic1")).all()

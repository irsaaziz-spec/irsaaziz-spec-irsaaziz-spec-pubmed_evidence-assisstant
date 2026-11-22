import json

import pytest
from fastapi.testclient import TestClient

from pubmed_rag.api import create_app
from pubmed_rag.config import Settings

Q = {"question": "Does topic9 treatment improve outcome?"}


@pytest.fixture()
def client(assistant, settings):
    with TestClient(create_app(settings, assistant)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["backend"] == "fake" and body["corpus_size"] == 80 and body["calibrated"]


def test_ask_returns_the_documented_schema(client):
    r = client.post("/ask", json=Q)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] in {"yes", "no", "maybe"} and len(body["evidence"]) == 5
    assert {"request_id", "probabilities", "timings_ms", "disclaimer", "corpus_size", "warning"} <= set(body)


@pytest.mark.parametrize("payload", [{"question": "hi"}, {"question": "x" * 501}, {**Q, "k": 0}, {**Q, "k": 11}, {}])
def test_validation_errors(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_feedback_requires_a_known_request(client, settings):
    assert client.post("/feedback", json={"request_id": "nope", "helpful": True}).status_code == 404
    rid = client.post("/ask", json=Q).json()["request_id"]
    assert client.post("/feedback", json={"request_id": rid, "helpful": False, "comment": "wrong"}).status_code == 200
    rows = [json.loads(line) for line in (settings.log_dir / "feedback.jsonl").read_text().splitlines()]
    assert rows[0]["request_id"] == rid and rows[0]["helpful"] is False


def test_request_log_does_not_store_the_question_by_default(client, settings):
    client.post("/ask", json=Q)
    entry = json.loads((settings.log_dir / "requests.jsonl").read_text().splitlines()[0])
    assert "question" not in entry and len(entry["question_hash"]) == 12 and entry["timings_ms"]["total"] >= 0


def test_question_logging_is_opt_in(assistant, tmp_path):
    s = Settings(log_dir=tmp_path / "q", log_questions=True)
    with TestClient(create_app(s, assistant)) as c:
        c.post("/ask", json=Q)
    assert json.loads((s.log_dir / "requests.jsonl").read_text().splitlines()[0])["question"] == Q["question"]


def test_stats_and_explain(client):
    assert client.get("/stats").json()["requests_total"] == 0
    r = client.post("/ask", json={**Q, "explain": True}).json()
    assert r["explanation"]
    stats = client.get("/stats").json()
    assert stats["requests_total"] == 1 and stats["p50_ms"] is not None


def test_startup_fails_loudly_without_an_index(tmp_path):
    app = create_app(Settings(index_dir=tmp_path / "missing", log_dir=tmp_path / "l"))
    with pytest.raises(FileNotFoundError), TestClient(app):
        pass


def test_build_assistant_falls_back_to_bm25_when_embeddings_cannot_be_used(index_dir, tmp_path, monkeypatch):
    import pubmed_rag.embedding as embedding
    from pubmed_rag.api import build_assistant

    def boom(*args, **kwargs):
        raise ImportError("no sentence-transformers")

    monkeypatch.setattr(embedding, "SentenceTransformerEmbedder", boom)
    a = build_assistant(Settings(index_dir=index_dir, backend="none", retrieval_mode="dense", log_dir=tmp_path / "l"))
    assert a.mode == "bm25" and a.backend.name == "none" and a.calibrator is None
    assert a.ask("Does topic4 treatment improve outcome?")["evidence"][0]["doc_id"] == "1004"


def test_build_assistant_without_embeddings_file(index_dir, tmp_path):
    from pubmed_rag.api import build_assistant

    (index_dir / "embeddings.npy").unlink()
    a = build_assistant(Settings(index_dir=index_dir, retrieval_mode="hybrid", log_dir=tmp_path / "l"))
    assert a.mode == "bm25"

import json

import numpy as np
import pytest

from pubmed_rag.evaluate import retrieval_metrics
from pubmed_rag.retrieval import Corpus, Retriever


def test_bm25_finds_the_gold_abstract(retriever, fixture_data):
    _, questions = fixture_data
    m = retrieval_metrics(retriever, questions, "bm25", k=5)
    assert m["hit_at_k"] >= 0.95 and m["mrr"] >= 0.9 and m["n"] == len(questions)


def test_dense_finds_the_gold_abstract(retriever, fixture_data):
    _, questions = fixture_data
    assert retrieval_metrics(retriever, questions, "dense", k=5)["hit_at_k"] >= 0.8


def test_hybrid_returns_k_unique_ranked_docs(retriever):
    hits = retriever.search("Does topic7 treatment improve outcome?", k=5, mode="hybrid")
    assert len(hits) == 5 and len({h.doc_id for h in hits}) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert hits[0].doc_id == "1007"


def test_scores_are_sorted_descending(retriever):
    for mode in ("dense", "bm25", "hybrid"):
        scores = [h.score for h in retriever.search("topic3 outcome", k=5, mode=mode)]
        assert scores == sorted(scores, reverse=True)


def test_k_larger_than_corpus_is_clamped(retriever):
    assert len(retriever.search("topic1", k=10_000, mode="dense")) == len(retriever.corpus)


def test_bm25_with_no_matching_terms_returns_nothing(retriever):
    assert retriever.search("zzzz qqqq", k=5, mode="bm25") == []


def test_dense_without_embedder_raises(corpus):
    with pytest.raises(RuntimeError):
        Retriever(corpus, embedder=None).search("topic1", mode="dense")


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.search("topic1", mode="magic")


def test_corpus_load_errors(tmp_path, index_dir):
    with pytest.raises(FileNotFoundError):
        Corpus.load(tmp_path / "nowhere")
    np.save(index_dir / "embeddings.npy", np.zeros((3, 8), dtype="float32"))
    with pytest.raises(ValueError):
        Corpus.load(index_dir)


def test_corpus_without_embeddings_loads(index_dir):
    (index_dir / "embeddings.npy").unlink()
    c = Corpus.load(index_dir)
    assert c.embeddings is None and len(c) == 80
    assert json.loads((index_dir / "meta.json").read_text())["n_docs"] == 80

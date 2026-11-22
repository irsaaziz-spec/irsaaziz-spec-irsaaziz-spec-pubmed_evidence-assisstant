"""Shared fixtures: a tiny synthetic corpus (80 'abstracts', one unique topic token each) and fake model backends.
These are for testing the engineering, not for measuring model quality."""

from __future__ import annotations

import hashlib
import random
import re

import numpy as np
import pytest

from pubmed_rag.build import write_index
from pubmed_rag.calibrate import Calibrator
from pubmed_rag.config import Settings
from pubmed_rag.pipeline import Assistant
from pubmed_rag.retrieval import Corpus, Retriever

FILL = "patients study group clinical analysis significant trial cohort effect baseline measure data".split()


def make_fixture(n: int = 80, seed: int = 0):
    rng = random.Random(seed)
    docs, questions = [], []
    for i in range(n):
        body = " ".join(rng.choices(FILL, k=40))
        tail = " ".join(rng.choices(FILL, k=20))
        docs.append({"doc_id": str(1000 + i), "text": f"Background: topic{i} {body}\nResults: topic{i} {tail}"})
        questions.append({"pubid": str(1000 + i), "question": f"Does topic{i} treatment improve outcome?",
                          "decision": rng.choice(["yes", "no", "maybe"])})
    return docs, questions


class HashEmbedder:
    """Deterministic bag-of-words embedder (no model download)."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])


class FakeBackend:
    """Stands in for the LLM; records the prompts it receives."""

    name = "fake"
    available = True

    def __init__(self, logprobs=(-0.5, -1.5, -3.0)):
        self.logprobs = list(logprobs)
        self.score_calls: list[tuple[str, str]] = []
        self.generate_calls = 0

    def score_decision(self, system, user):
        self.score_calls.append((system, user))
        return self.logprobs, 123

    def generate(self, system, user, max_new_tokens=100):
        self.generate_calls += 1
        return "The abstracts report a modest benefit.", 100, 12


@pytest.fixture()
def fixture_data():
    return make_fixture()


@pytest.fixture()
def index_dir(tmp_path, fixture_data):
    docs, questions = fixture_data
    write_index(tmp_path / "index", docs, questions, HashEmbedder().encode_passages([d["text"] for d in docs]),
                {"embed_model": "hash"})
    return tmp_path / "index"


@pytest.fixture()
def corpus(index_dir):
    return Corpus.load(index_dir)


@pytest.fixture()
def retriever(corpus):
    return Retriever(corpus, HashEmbedder())


@pytest.fixture()
def identity_calibrator():
    """Maps the (normalised) log-probs straight to a softmax over yes/no/maybe."""
    return Calibrator(["yes", "no", "maybe"], np.eye(3).tolist(), [0.0, 0.0, 0.0])


@pytest.fixture()
def settings(tmp_path):
    return Settings(index_dir=tmp_path / "index", log_dir=tmp_path / "logs", retrieval_mode="dense")


@pytest.fixture()
def assistant(retriever, identity_calibrator, settings):
    return Assistant(retriever, FakeBackend(), identity_calibrator, settings)

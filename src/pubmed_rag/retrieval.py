"""Corpus index and retrieval: BM25, dense (cosine over precomputed embeddings) and hybrid (RRF)."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

MODES = ("dense", "bm25", "hybrid")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class QueryEmbedder(Protocol):
    def encode_query(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class Hit:
    doc_id: str
    text: str
    score: float
    rank: int


@dataclass
class Corpus:
    doc_ids: list[str]
    texts: list[str]
    embeddings: np.ndarray | None = None
    meta: dict | None = None

    @classmethod
    def load(cls, index_dir: str | Path) -> Corpus:
        index_dir = Path(index_dir)
        docs_file = index_dir / "docs.jsonl"
        if not docs_file.exists():
            raise FileNotFoundError(
                f"{docs_file} not found. Build the index first: python scripts/build_index.py --out {index_dir}")
        rows = [json.loads(line) for line in docs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        emb_file = index_dir / "embeddings.npy"
        embeddings = np.load(emb_file).astype("float32") if emb_file.exists() else None
        if embeddings is not None and len(embeddings) != len(rows):
            raise ValueError(f"{emb_file} has {len(embeddings)} rows but docs.jsonl has {len(rows)}")
        meta_file = index_dir / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else None
        return cls([r["doc_id"] for r in rows], [r["text"] for r in rows], embeddings, meta)

    def __len__(self) -> int:
        return len(self.doc_ids)


class Retriever:
    def __init__(self, corpus: Corpus, embedder: QueryEmbedder | None = None, rrf_pool: int = 30):
        self.corpus = corpus
        self.embedder = embedder
        self.rrf_pool = rrf_pool
        self._bm25 = BM25Okapi([tokenize(t) for t in corpus.texts])

    @property
    def has_dense(self) -> bool:
        return self.corpus.embeddings is not None and self.embedder is not None

    def search(self, query: str, k: int = 5, mode: str = "dense") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        k = max(1, min(k, len(self.corpus)))
        ranking = {"dense": self._dense, "bm25": self._bm25_rank, "hybrid": self._hybrid}[mode](query, k)
        return [Hit(self.corpus.doc_ids[i], self.corpus.texts[i], float(s), r + 1) for r, (i, s) in enumerate(ranking)]

    def _dense(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self.has_dense:
            raise RuntimeError("dense search needs embeddings.npy and an embedder; use mode='bm25' instead")
        scores = self.corpus.embeddings @ np.asarray(self.embedder.encode_query(query), dtype="float32")
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def _bm25_rank(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def _hybrid(self, query: str, k: int) -> list[tuple[int, float]]:
        pool = max(self.rrf_pool, k)
        fused: dict[int, float] = defaultdict(float)
        for ranking in (self._dense(query, min(pool, len(self.corpus))), self._bm25_rank(query, pool)):
            for rank, (i, _) in enumerate(ranking):
                fused[i] += 1.0 / (60 + rank + 1)
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]

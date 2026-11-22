"""Retrieval evaluation: hit@k and MRR, where the gold document for a question is its own PubMed ID."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from .retrieval import Retriever


def retrieval_metrics(retriever: Retriever, questions: Sequence[dict], mode: str, k: int = 5) -> dict:
    t0 = time.perf_counter()
    ranks: list[int | None] = []
    for q in questions:
        ids = [h.doc_id for h in retriever.search(q["question"], k, mode)]
        ranks.append(ids.index(str(q["pubid"])) + 1 if str(q["pubid"]) in ids else None)
    n = len(ranks)
    return {"mode": mode, "k": k, "n": n,
            "hit_at_k": float(np.mean([r is not None for r in ranks])),
            "mrr": float(np.mean([1 / r if r else 0.0 for r in ranks])),
            "ms_per_query": (time.perf_counter() - t0) / max(n, 1) * 1000}

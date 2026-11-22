"""Query/passage embedding with sentence-transformers (imported lazily: the core API does not need torch)."""

from __future__ import annotations

import threading

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge-*-en-v1.5 query instruction


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        if torch.cuda.is_available():
            self.model.half()
        self._lock = threading.Lock()

    def encode_query(self, text: str) -> np.ndarray:
        with self._lock:
            vec = self.model.encode([QUERY_PREFIX + text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype="float32")

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        with self._lock:
            vecs = self.model.encode(texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True)
        return np.asarray(vecs, dtype="float32")

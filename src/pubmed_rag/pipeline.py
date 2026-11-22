"""The assistant: retrieve -> (optional) LLM decision -> calibrate -> response."""

from __future__ import annotations

import re
import time
import uuid

from .calibrate import Calibrator, softmax_from_logprobs
from .config import Settings
from .labels import CLASSES
from .llm import LLMBackend
from .prompts import DISCLAIMER, SYSTEM_EXPLAIN, SYSTEM_RAG, build_user_prompt
from .retrieval import Retriever

PERSONAL_ADVICE = re.compile(
    r"\b(should i|can i (take|use|stop|give)|do i need|is it safe for me|my (doctor|child|baby|son|daughter|"
    r"mother|father|dose|medication|symptoms?|pregnan\w*))\b", re.IGNORECASE)
PERSONAL_ADVICE_WARNING = (
    "This looks like a personal medical question. This tool only summarises published research and cannot "
    "give medical advice; please talk to a qualified clinician.")
NO_EVIDENCE_WARNING = "No relevant abstracts were found in the corpus, so no answer is given."


class Assistant:
    def __init__(self, retriever: Retriever, backend: LLMBackend, calibrator: Calibrator | None, settings: Settings):
        self.retriever = retriever
        self.backend = backend
        self.calibrator = calibrator
        self.settings = settings
        # dense search needs embeddings + an embedder; otherwise fall back to BM25 and say so
        self.mode = settings.retrieval_mode
        if self.mode in ("dense", "hybrid") and not retriever.has_dense:
            self.mode = "bm25"

    def ask(self, question: str, k: int | None = None, explain: bool = False) -> dict:
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k or self.settings.k, self.mode)
        t_retrieval = time.perf_counter()

        warning = PERSONAL_ADVICE_WARNING if PERSONAL_ADVICE.search(question) else None
        decision = probs = explanation = None
        calibrated = low_confidence = False
        scoring_ms = explain_ms = 0.0

        if not hits:
            warning = NO_EVIDENCE_WARNING
        elif self.backend.available:
            t = time.perf_counter()
            logprobs, _ = self.backend.score_decision(SYSTEM_RAG, build_user_prompt(question, hits))
            if self.calibrator is not None:
                probs, calibrated = self.calibrator.predict_proba(logprobs), True
            else:
                probs = softmax_from_logprobs(logprobs)
            decision = max(CLASSES, key=lambda c: probs[c])
            low_confidence = probs[decision] < self.settings.low_confidence_threshold
            scoring_ms = (time.perf_counter() - t) * 1000
            if explain:
                t = time.perf_counter()
                explanation, _, _ = self.backend.generate(SYSTEM_EXPLAIN, build_user_prompt(question, hits), 120)
                explain_ms = (time.perf_counter() - t) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        return {
            "request_id": uuid.uuid4().hex[:12],
            "decision": decision,
            "probabilities": probs,
            "calibrated": calibrated,
            "low_confidence": low_confidence,
            "evidence": [{"rank": h.rank, "doc_id": h.doc_id, "score": round(h.score, 4), "text": h.text} for h in hits],
            "explanation": explanation,
            "warning": warning,
            "disclaimer": DISCLAIMER,
            "backend": self.backend.name,
            "retrieval_mode": self.mode,
            "corpus_size": len(self.retriever.corpus),
            "timings_ms": {"retrieval": round((t_retrieval - t0) * 1000, 1), "scoring": round(scoring_ms, 1),
                           "explain": round(explain_ms, 1), "total": round(total_ms, 1)},
        }

"""FastAPI service.   uvicorn pubmed_rag.api:create_app --factory --port 8000"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from . import __version__
from .calibrate import Calibrator
from .config import Settings
from .llm import LLMBackend, LocalHFBackend, NullBackend
from .pipeline import Assistant
from .retrieval import Corpus, Retriever
from .schemas import AskRequest, AskResponse, FeedbackRequest
from .telemetry import JsonlLogger, RollingStats

log = logging.getLogger("pubmed_rag")


def build_assistant(settings: Settings) -> Assistant:
    """Load the index, embedder, LLM backend and calibrator described by `settings`."""
    corpus = Corpus.load(settings.index_dir)

    embedder = None
    if settings.retrieval_mode in ("dense", "hybrid") and corpus.embeddings is not None:
        try:
            from .embedding import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(settings.embed_model)
        except ImportError:
            log.warning("sentence-transformers not installed; falling back to BM25 retrieval")
    elif settings.retrieval_mode in ("dense", "hybrid"):
        log.warning("index has no embeddings.npy; falling back to BM25 retrieval")

    backend: LLMBackend = NullBackend()
    if settings.backend == "local":
        backend = LocalHFBackend(settings.llm_id, settings.load_4bit)

    calibrator = None
    if backend.available and settings.calibrator_file.exists():
        calibrator = Calibrator.from_json(settings.calibrator_file, settings.calibrator_variant)
    elif backend.available:
        log.warning("no calibrator at %s; using the LLM's raw probabilities", settings.calibrator_file)

    return Assistant(Retriever(corpus, embedder), backend, calibrator, settings)


def create_app(settings: Settings | None = None, assistant: Assistant | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logger = JsonlLogger(settings.log_dir, settings.log_questions)
    stats = RollingStats()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.assistant = assistant or build_assistant(settings)
        yield

    app = FastAPI(title="PubMed Evidence Assistant", version=__version__, lifespan=lifespan,
                  description="Yes/no/maybe answers to research questions, grounded in retrieved PubMed abstracts.")

    @app.get("/health")
    def health() -> dict:
        a: Assistant = app.state.assistant
        return {"status": "ok", "version": __version__, "backend": a.backend.name, "retrieval_mode": a.mode,
                "calibrated": a.calibrator is not None, "corpus_size": len(a.retriever.corpus)}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict:
        a: Assistant = app.state.assistant
        result = a.ask(req.question.strip(), req.k, req.explain)
        stats.add(result["request_id"], result["timings_ms"]["total"])
        logger.log_request(result["request_id"], req.question, {
            "decision": result["decision"], "backend": result["backend"], "mode": result["retrieval_mode"],
            "k": len(result["evidence"]), "timings_ms": result["timings_ms"], "warning": result["warning"],
            "top_doc": result["evidence"][0]["doc_id"] if result["evidence"] else None})
        return result

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        if not stats.knows(req.request_id):
            raise HTTPException(status_code=404, detail="unknown request_id")
        logger.log_feedback(req.request_id, req.helpful, req.comment)
        return {"status": "recorded"}

    @app.get("/stats")
    def get_stats() -> dict:
        return stats.summary()

    return app

"""Request / response models for the HTTP API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int | None = Field(default=None, ge=1, le=10, description="abstracts to retrieve (default: server setting)")
    explain: bool = Field(default=False, description="also generate a short evidence summary (slower)")


class Evidence(BaseModel):
    rank: int
    doc_id: str = Field(description="PubMed ID")
    score: float
    text: str


class AskResponse(BaseModel):
    request_id: str
    decision: Literal["yes", "no", "maybe"] | None = Field(
        description="null when the server runs retrieval-only or nothing relevant was found")
    probabilities: dict[str, float] | None
    calibrated: bool
    low_confidence: bool
    evidence: list[Evidence]
    explanation: str | None
    warning: str | None
    disclaimer: str
    backend: str
    retrieval_mode: str
    corpus_size: int
    timings_ms: dict[str, float]


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    helpful: bool
    comment: str | None = Field(default=None, max_length=500)

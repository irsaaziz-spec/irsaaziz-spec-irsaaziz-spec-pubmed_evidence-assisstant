"""Runtime configuration, read from PMRAG_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKENDS = ("none", "local")
RETRIEVAL_MODES = ("dense", "bm25", "hybrid")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    index_dir: Path = Path("index")
    backend: str = "none"                  # "local" = open LLM on this machine, "none" = retrieval-only
    retrieval_mode: str = "dense"          # the configuration that was evaluated
    k: int = 5                             # abstracts sent to the LLM (also the evaluated setting)
    llm_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_4bit: bool = False
    embed_model: str = "BAAI/bge-base-en-v1.5"
    calibrator_path: Path | None = None    # defaults to <index_dir>/calibrator.json
    calibrator_variant: str = "unweighted"  # "unweighted" (accuracy) or "balanced" (macro-F1)
    low_confidence_threshold: float = 0.5
    log_dir: Path = Path("logs")
    log_questions: bool = False            # privacy: off by default, only a hash of each question is stored

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {self.retrieval_mode!r}")
        if self.k < 1:
            raise ValueError("k must be >= 1")

    @property
    def calibrator_file(self) -> Path:
        return self.calibrator_path or self.index_dir / "calibrator.json"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        defaults = cls()
        return cls(
            index_dir=Path(env("PMRAG_INDEX_DIR", str(defaults.index_dir))),
            backend=env("PMRAG_BACKEND", defaults.backend),
            retrieval_mode=env("PMRAG_RETRIEVAL_MODE", defaults.retrieval_mode),
            k=int(env("PMRAG_K", str(defaults.k))),
            llm_id=env("PMRAG_LLM_ID", defaults.llm_id),
            load_4bit=_env_bool("PMRAG_LOAD_4BIT", defaults.load_4bit),
            embed_model=env("PMRAG_EMBED_MODEL", defaults.embed_model),
            calibrator_path=Path(env("PMRAG_CALIBRATOR")) if env("PMRAG_CALIBRATOR") else None,
            calibrator_variant=env("PMRAG_CALIBRATOR_VARIANT", defaults.calibrator_variant),
            low_confidence_threshold=float(env("PMRAG_LOW_CONFIDENCE", str(defaults.low_confidence_threshold))),
            log_dir=Path(env("PMRAG_LOG_DIR", str(defaults.log_dir))),
            log_questions=_env_bool("PMRAG_LOG_QUESTIONS", defaults.log_questions),
        )

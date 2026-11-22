"""Build the retrieval index from PubMedQA: the 1,000 expert-labelled abstracts plus unlabeled distractors."""

from __future__ import annotations

import json
import time
from pathlib import Path


def to_abstract(ctx: dict) -> str:
    """PubMedQA 'context' = parallel lists (section text, section labels). The conclusion is NOT included."""
    parts = []
    for label, text in zip(ctx["labels"], ctx["contexts"], strict=False):
        parts.append(f"{str(label).title()}: {text}" if label else text)
    return "\n".join(parts)


def load_pubmedqa(n_distractors: int = 4000) -> tuple[list[dict], list[dict]]:
    from datasets import load_dataset

    labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    questions = [{"pubid": str(r["pubid"]), "question": r["question"], "decision": r["final_decision"]} for r in labeled]
    pool = {str(r["pubid"]): to_abstract(r["context"]) for r in labeled}
    extra = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled", split="train", streaming=True).take(n_distractors)
    for r in extra:
        pool.setdefault(str(r["pubid"]), to_abstract(r["context"]))
    return questions, [{"doc_id": k, "text": v} for k, v in pool.items()]


def write_index(out_dir: str | Path, docs: list[dict], questions: list[dict], embeddings=None, meta: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "docs.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n", encoding="utf-8")
    (out / "questions.jsonl").write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")
    if embeddings is not None:
        import numpy as np

        np.save(out / "embeddings.npy", np.asarray(embeddings, dtype="float32"))
    (out / "meta.json").write_text(json.dumps({"n_docs": len(docs), "n_questions": len(questions),
                                               "has_embeddings": embeddings is not None, **(meta or {})}, indent=2), encoding="utf-8")


def build_index(out_dir: str | Path, n_distractors: int = 4000, embed: bool = True,
                embed_model: str = "BAAI/bge-base-en-v1.5") -> None:
    questions, docs = load_pubmedqa(n_distractors)
    embeddings = None
    if embed:
        from .embedding import SentenceTransformerEmbedder

        embeddings = SentenceTransformerEmbedder(embed_model).encode_passages([d["text"] for d in docs])
    write_index(out_dir, docs, questions, embeddings,
                {"source": "qiaojin/PubMedQA (pqa_labeled + pqa_unlabeled distractors)", "n_distractors": n_distractors,
                 "embed_model": embed_model if embed else None, "built_at": time.strftime("%Y-%m-%dT%H:%M:%S")})

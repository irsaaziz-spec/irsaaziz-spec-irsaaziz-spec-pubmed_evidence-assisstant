"""Request logging and rolling latency statistics (AIOps-lite)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


def question_hash(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()[:12]


class JsonlLogger:
    """Append-only JSONL logs. Questions are stored as a short hash unless log_questions=True."""

    def __init__(self, log_dir: str | Path, log_questions: bool = False):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_questions = log_questions
        self._lock = threading.Lock()

    def _append(self, name: str, record: dict) -> None:
        line = json.dumps({"ts": time.time(), **record}, ensure_ascii=False)
        with self._lock, (self.dir / name).open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_request(self, request_id: str, question: str, record: dict) -> None:
        entry = {"request_id": request_id, "question_hash": question_hash(question), **record}
        if self.log_questions:
            entry["question"] = question
        self._append("requests.jsonl", entry)

    def log_feedback(self, request_id: str, helpful: bool, comment: str | None) -> None:
        self._append("feedback.jsonl", {"request_id": request_id, "helpful": helpful, "comment": comment})


class RollingStats:
    def __init__(self, maxlen: int = 1000):
        self._total_ms: deque[float] = deque(maxlen=maxlen)
        self._ids: deque[str] = deque(maxlen=maxlen)
        self._count = 0
        self._lock = threading.Lock()

    def add(self, request_id: str, total_ms: float) -> None:
        with self._lock:
            self._total_ms.append(total_ms)
            self._ids.append(request_id)
            self._count += 1

    def knows(self, request_id: str) -> bool:
        with self._lock:
            return request_id in self._ids

    def summary(self) -> dict:
        with self._lock:
            values = list(self._total_ms)
            count = self._count
        if not values:
            return {"requests_total": count, "window": 0, "p50_ms": None, "p95_ms": None}
        return {"requests_total": count, "window": len(values),
                "p50_ms": round(float(np.percentile(values, 50)), 1), "p95_ms": round(float(np.percentile(values, 95)), 1)}

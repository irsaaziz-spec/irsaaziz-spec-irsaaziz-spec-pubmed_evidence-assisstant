"""Calibrated decision head: logistic regression over the LLM's yes/no/maybe log-probabilities.

Inference is pure numpy (no sklearn needed in the API); sklearn is only used to fit.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from .labels import CLASSES


def normalise_logprobs(logprobs: Sequence[float]) -> np.ndarray:
    """Renormalise log-probs over the three answer tokens so they form a distribution."""
    x = np.asarray(logprobs, dtype=float)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


class Calibrator:
    def __init__(self, classes: Sequence[str], coef, intercept, feature_order: Sequence[str] = CLASSES):
        self.classes = list(classes)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = np.asarray(intercept, dtype=float)
        self.feature_order = tuple(feature_order)
        if self.coef.shape != (len(self.classes), len(self.feature_order)):
            raise ValueError(f"coef shape {self.coef.shape} does not match classes/features")
        if tuple(self.feature_order) != tuple(CLASSES):
            raise ValueError(f"feature_order must be {CLASSES}, got {self.feature_order}")

    def predict_proba(self, logprobs: Sequence[float]) -> dict[str, float]:
        z = self.coef @ normalise_logprobs(logprobs) + self.intercept
        z = z - z.max()
        p = np.exp(z) / np.exp(z).sum()
        return {c: float(v) for c, v in zip(self.classes, p, strict=True)}

    @classmethod
    def from_dict(cls, payload: dict, variant: str) -> Calibrator:
        try:
            v = payload["variants"][variant]
        except KeyError as exc:
            raise ValueError(f"calibrator variant {variant!r} not found; have {list(payload.get('variants', {}))}") from exc
        return cls(v["classes"], v["coef"], v["intercept"], payload.get("feature_order", CLASSES))

    @classmethod
    def from_json(cls, path: str | Path, variant: str = "unweighted") -> Calibrator:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")), variant)


def softmax_from_logprobs(logprobs: Sequence[float]) -> dict[str, float]:
    """Uncalibrated fallback: the model's own distribution over yes/no/maybe."""
    p = np.exp(normalise_logprobs(logprobs))
    return {c: float(v) for c, v in zip(CLASSES, p, strict=True)}


def fit_from_jsonl(paths: Iterable[str | Path], out_path: str | Path) -> dict:
    """Fit both calibrator variants from score files written by the evaluation notebook
    (one JSON object per line with keys 'gold' and 'lp') and save them as JSON."""
    from sklearn.linear_model import LogisticRegression  # fit-time dependency only

    rows = []
    for p in paths:
        rows += [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) < 30:
        raise ValueError(f"only {len(rows)} training rows; need a few hundred for a stable calibrator")
    x = np.array([normalise_logprobs(r["lp"]) for r in rows])
    y = np.array([r["gold"] for r in rows])

    variants = {}
    for name, class_weight in (("unweighted", None), ("balanced", "balanced")):
        clf = LogisticRegression(class_weight=class_weight, C=1.0, max_iter=2000).fit(x, y)
        variants[name] = {"classes": [str(c) for c in clf.classes_], "coef": clf.coef_.tolist(),
                          "intercept": clf.intercept_.tolist()}
    payload = {"feature_order": list(CLASSES), "n_train": len(rows), "variants": variants}
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

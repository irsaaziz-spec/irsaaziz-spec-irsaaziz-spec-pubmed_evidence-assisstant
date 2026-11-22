import json

import numpy as np
import pytest

from pubmed_rag.calibrate import Calibrator, fit_from_jsonl, normalise_logprobs, softmax_from_logprobs

pytest.importorskip("sklearn")


def _write_scores(path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    labels = ["yes", "no", "maybe"]
    rows = []
    for _ in range(n):
        gold = rng.choice(labels, p=[0.55, 0.34, 0.11])
        lp = rng.normal(size=3)
        lp[labels.index(gold)] += 1.0
        rows.append({"pubid": "1", "gold": str(gold), "lp": lp.tolist()})
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


def test_numpy_inference_matches_sklearn(tmp_path):
    from sklearn.linear_model import LogisticRegression

    scores = tmp_path / "scores.jsonl"
    rows = _write_scores(scores)
    out = tmp_path / "cal.json"
    fit_from_jsonl([scores], out)

    x = np.array([normalise_logprobs(r["lp"]) for r in rows])
    y = np.array([r["gold"] for r in rows])
    for variant, weight in (("unweighted", None), ("balanced", "balanced")):
        clf = LogisticRegression(class_weight=weight, C=1.0, max_iter=2000).fit(x, y)
        cal = Calibrator.from_json(out, variant)
        for row, expected in zip(rows[:25], clf.predict_proba(x[:25]), strict=True):
            got = cal.predict_proba(row["lp"])
            assert sum(got.values()) == pytest.approx(1.0)
            assert [got[c] for c in clf.classes_] == pytest.approx(expected.tolist(), abs=1e-9)


def test_fit_requires_enough_rows(tmp_path):
    scores = tmp_path / "s.jsonl"
    _write_scores(scores, n=10)
    with pytest.raises(ValueError):
        fit_from_jsonl([scores], tmp_path / "c.json")


def test_unknown_variant_and_bad_feature_order(tmp_path):
    scores = tmp_path / "s.jsonl"
    _write_scores(scores)
    fit_from_jsonl([scores], tmp_path / "c.json")
    with pytest.raises(ValueError):
        Calibrator.from_json(tmp_path / "c.json", "nope")
    with pytest.raises(ValueError):
        Calibrator(["yes", "no", "maybe"], np.eye(3), [0, 0, 0], feature_order=("no", "yes", "maybe"))


def test_normalised_logprobs_form_a_distribution():
    assert np.exp(normalise_logprobs([-1.0, -2.0, -5.0])).sum() == pytest.approx(1.0)
    p = softmax_from_logprobs([-0.1, -3.0, -6.0])
    assert max(p, key=p.get) == "yes" and sum(p.values()) == pytest.approx(1.0)

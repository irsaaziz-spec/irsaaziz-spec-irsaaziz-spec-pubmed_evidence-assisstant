"""Retrieval evaluation over the questions in an index directory (gold = the question's own PubMed ID).

   python scripts/eval_retrieval.py --index index --k 5 --modes bm25 dense hybrid
   python scripts/eval_retrieval.py --index index --modes bm25 --min-hit 0.9     # exits 1 below the threshold
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pubmed_rag.evaluate import retrieval_metrics  # noqa: E402
from pubmed_rag.retrieval import Corpus, Retriever  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="index")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-hit", type=float, default=None, help="fail if hit@k of any evaluated mode is below this")
    ap.add_argument("--out", default=None, help="write the metrics as JSON here")
    args = ap.parse_args()

    corpus = Corpus.load(args.index)
    questions = [json.loads(line) for line in (Path(args.index) / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = questions[: args.limit] if args.limit else questions

    embedder = None
    if any(m in ("dense", "hybrid") for m in args.modes) and corpus.embeddings is not None:
        try:
            from pubmed_rag.embedding import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder((corpus.meta or {}).get("embed_model") or "BAAI/bge-base-en-v1.5")
        except ImportError as exc:
            print(f"dense/hybrid unavailable ({exc}); install requirements-ml.txt to evaluate them")
    retriever = Retriever(corpus, embedder)

    results = []
    for mode in args.modes:
        if mode in ("dense", "hybrid") and not retriever.has_dense:
            print(f"skipping {mode}: no embeddings/embedder")
            continue
        results.append(retrieval_metrics(retriever, questions, mode, args.k))
        r = results[-1]
        print(f"{mode:7s} hit@{args.k}={r['hit_at_k']:.3f}  mrr={r['mrr']:.3f}  {r['ms_per_query']:.1f} ms/query  (n={r['n']})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.min_hit is not None and any(r["hit_at_k"] < args.min_hit for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

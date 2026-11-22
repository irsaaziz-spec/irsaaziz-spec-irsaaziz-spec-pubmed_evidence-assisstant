"""Build the retrieval index from PubMedQA.   python scripts/build_index.py --out index"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pubmed_rag.build import build_index  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="index")
    ap.add_argument("--distractors", type=int, default=4000, help="extra unlabeled abstracts (4000 matches the evaluation)")
    ap.add_argument("--no-embeddings", action="store_true", help="BM25-only index (no torch needed)")
    ap.add_argument("--embed-model", default="BAAI/bge-base-en-v1.5")
    args = ap.parse_args()
    build_index(args.out, args.distractors, embed=not args.no_embeddings, embed_model=args.embed_model)
    print(f"index written to {args.out}/")


if __name__ == "__main__":
    main()

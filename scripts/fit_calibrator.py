"""Fit the calibrated decision head from score files written by the evaluation notebook.

   python scripts/fit_calibrator.py --scores "results/score_train_rag_whole_*.jsonl" --out index/calibrator.json
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pubmed_rag.calibrate import fit_from_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="glob of JSONL files with 'gold' and 'lp' per line")
    ap.add_argument("--out", default="index/calibrator.json")
    args = ap.parse_args()
    files = sorted(glob.glob(args.scores), key=lambda p: Path(p).stat().st_mtime)
    if not files:
        sys.exit(f"no files match {args.scores!r}")
    payload = fit_from_jsonl(files[-1:], args.out)  # newest file only: older ones may come from other prompts/models
    print(f"fitted on {payload['n_train']} rows from {files[-1]} -> {args.out}")


if __name__ == "__main__":
    main()

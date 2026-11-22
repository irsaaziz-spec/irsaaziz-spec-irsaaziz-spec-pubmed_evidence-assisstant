import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_retrieval.py"


def _run(index_dir, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), "--index", str(index_dir), *extra],
                          capture_output=True, text=True, timeout=120)


def test_eval_script_reports_and_gates(index_dir):
    ok = _run(index_dir, "--modes", "bm25", "--min-hit", "0.9")
    assert ok.returncode == 0 and "bm25" in ok.stdout and "hit@5" in ok.stdout
    assert _run(index_dir, "--modes", "bm25", "--min-hit", "1.1").returncode == 1


def test_eval_script_skips_dense_when_unavailable(index_dir):
    out = _run(index_dir, "--modes", "bm25", "dense")  # no embedder installed in CI -> dense is skipped or runs
    assert out.returncode == 0 and "bm25" in out.stdout

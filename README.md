# PubMed Evidence Assistant

Answers biomedical **research** questions with a **yes / no / maybe** grounded in retrieved PubMed abstracts, shows the abstracts it used (with links), and reports calibrated probabilities instead of a bare label. It runs an open 4B model (`Qwen3-4B-Instruct-2507`) on a single free GPU, so there is no API key and no per-query cost.

> **Not medical advice.** It summarises a fixed set of published abstracts, can be wrong, and says so in every response.

<!-- Add a 60-90 second demo GIF/video here. Then add the CI badge:
[![ci](https://github.com/<your-username>/pubmed-evidence-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/pubmed-evidence-assistant/actions)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/pubmed-evidence-assistant/blob/main/notebooks/03_serve_on_colab.ipynb) -->

## What I found (and what I did about it)

I built the evaluation before the product, on [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) (1,000 expert-labelled questions, MIT licence), with a 5,000-abstract search pool (the 1,000 gold abstracts plus 4,000 distractors).

1. **Retrieval was not the bottleneck.** Dense search finds the right abstract in the top 5 for ~99% of 1,000 questions, and handing the model the gold abstract (an "oracle") was not distinguishably better than retrieval on 150 dev questions.
2. **The LLM's own yes/no/maybe is badly biased.** With no evidence it said "maybe" to most questions. So instead of parsing generated text, the service reads the model's *probabilities* for the three answers in a single forward pass and passes them through a small **logistic-regression calibrator** fitted on 300 held-out questions.
3. **Context matters more than I expected.** Whole abstracts beat 1,000-character chunks, with similar retrieval hit-rates, at ~1.9x the prompt tokens.
4. **Most remaining errors are not retrieval errors**: in 56 of 58 wrong dev answers (class-balanced head, whole abstracts) the gold abstract had been retrieved. Reading the errors points at hedged conclusions, non-yes/no questions, and noisy labels.

### Results

**Retrieval** (all 1,000 questions, 5,000-abstract pool, k=5; gold = the question's own abstract)

| Method | hit@5 | MRR |
|---|---|---|
| Dense (bge-base-en-v1.5), whole abstracts | 0.991 | 0.983 |
| Hybrid (BM25 + dense, RRF) | 0.987 | 0.969 |
| BM25 | 0.967 | 0.931 |

Chunking (500 / 1,000 characters) did not change dense hit@5 (0.990-0.991).

**Answering** (dev split, n=150, stratified; 95% bootstrap CIs; accuracy / macro-F1)

| System | Accuracy | Macro-F1 |
|---|---|---|
| Majority class ("yes") | 0.553 | 0.237 |
| Closed-book, calibrated, accuracy-tuned (no retrieval) | 0.547 [0.47, 0.63] | 0.277 [0.23, 0.33] |
| RAG, 1,000-char chunks, calibrated, accuracy-tuned | 0.673 [0.59, 0.75] | 0.460 [0.41, 0.51] |
| RAG, whole abstracts, raw model argmax | 0.587 [0.51, 0.67] | 0.477 [0.40, 0.56] |
| **RAG, whole abstracts, calibrated (accuracy-tuned)** | **0.733 [0.67, 0.80]** | 0.510 [0.46, 0.55] |
| RAG, whole abstracts, calibrated (class-balanced) | 0.613 [0.54, 0.69] | 0.557 [0.48, 0.63] |
| Oracle (gold abstract), calibrated | 0.740 [0.67, 0.81] | 0.515 [0.47, 0.56] |

Paired comparisons on the same questions (95% CI excludes 0): chunk-RAG over closed-book, macro-F1 +0.18 [0.10, 0.26] and whole abstracts over chunks, accuracy +0.13 [0.05, 0.20] (both with the class-balanced head); accuracy-tuned calibration over raw argmax for whole-abstract RAG, accuracy +0.15 [0.07, 0.23]. **Not** distinguishable: oracle vs whole-abstract RAG (+0.05 [-0.02, 0.11]).

**Held-out test split (n=150):** _TODO: run notebook 02 with `EVAL_SPLIT = "test"` once and paste the table here._ Dev was used to pick the configuration, so the dev numbers above are optimistic.

**Latency:** generation-based runs on a Colab T4 had p50 1.6 s (chunks) to 2.1 s (whole abstracts). The service uses a single forward pass instead; measure it with `GET /stats` (_TODO: paste your numbers_).

## Architecture


* `GET /health`, `POST /ask`, `POST /feedback`, `GET /stats`; interactive docs at `/docs`.
* **Swappable backend:** `PMRAG_BACKEND=local` (open LLM on the machine) or `none` (retrieval-only: returns the cited abstracts, runs on any CPU, no torch).
* A request is serialised through one lock around the GPU; `/ask` returns decision, probabilities, evidence, warnings and per-stage timings.

## Quickstart

**Free GPU demo (Colab T4, public link):** open `notebooks/03_serve_on_colab.ipynb`, set the runtime to T4, run all cells. It builds the index, fits the calibrator from the saved training scores, starts the API, and launches the Gradio demo with a share link. The link lives only while the notebook runs.

**CPU only (retrieval-only API, Docker):**
```bash
python scripts/build_index.py --out index --no-embeddings   # needs `pip install datasets`
docker build -t pubmed-rag . && \
docker run -p 8000:8000 -e PMRAG_RETRIEVAL_MODE=bm25 -v "$PWD/index:/app/index" pubmed-rag
curl -s localhost:8000/ask -H 'content-type: application/json' \
     -d '{"question": "Does high blood pressure reduce the risk of chronic low back pain?"}'
```

**GPU machine:** `pip install -r requirements.txt -r requirements-ml.txt`, build the index (with embeddings), fit the calibrator (`scripts/fit_calibrator.py`), then `PMRAG_BACKEND=local uvicorn pubmed_rag.api:create_app --factory`. Configuration is via `PMRAG_*` environment variables (see `src/pubmed_rag/config.py`).

## How it is evaluated, and what guards the numbers

* Fixed stratified dev/test splits; the calibrator is fitted on a third, disjoint train split.
* PubMedQA's `long_answer` (the abstract's conclusion) is **never** put in a prompt; the corpus uses the abstract without its conclusion.
* `tests/test_prompts.py` checks that the service's prompt is byte-identical to the one the calibrator was fitted on. If they drift, the calibrator is silently invalid.
* `scripts/eval_retrieval.py` reproduces the retrieval table from the index and can fail a build below a threshold (`--min-hit`).
* CI runs `ruff`, `pytest` (retrieval, calibrator parity with scikit-learn, pipeline, API, background server, eval script) and a Docker build. The test corpus is synthetic and tests the engineering, not model quality.

## Limitations (please read)

* **Fixed corpus.** It searches 5,000 PubMed abstracts, not PubMed. Out-of-corpus questions retrieve irrelevant abstracts and still get a verdict; there is no relevance threshold yet.
* **"maybe" is not a real prediction.** The accuracy-tuned head never predicts it; the class-balanced head predicts it often with low precision. With 16 "maybe" questions in dev, per-class numbers are noisy. Low-confidence answers are flagged instead.
* **Label noise and ambiguity.** Several dev "errors" were arguably label problems or questions that are not yes/no. Treat accuracy as a rough guide.
* **Small samples.** n=150 gives roughly +/-8 points on accuracy. Roughly 22 paired comparisons were examined with no multiple-comparison correction.
* **Not comparable to the PubMedQA leaderboard** (different split and setting).
* **One 4B model**, evaluated on one corpus. No fine-tuning.
* **Privacy.** Requests are logged as a short hash by default; raw questions are stored only if `PMRAG_LOG_QUESTIONS=1`.

## Repo layout

```
src/pubmed_rag/   config, retrieval, llm backends, calibrator, pipeline, API, telemetry, evaluation, index build
app/demo.py       Gradio UI (talks to the API over HTTP)
scripts/          build_index, fit_calibrator, eval_retrieval
notebooks/        01 baseline + eval, 02 statistics + calibration, 03 serve on Colab
tests/            pytest suite (no model downloads needed)
```

## Roadmap

Cross-encoder reranker; a larger model in 4-bit vs this one (accuracy / latency / memory); QLoRA fine-tune on the labelled train split vs the calibrated head; a retrieval-relevance threshold for out-of-corpus questions; a React front end.

## License

MIT for the code. PubMedQA is MIT-licensed; abstracts are from PubMed.

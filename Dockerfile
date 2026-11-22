# CPU-friendly by default: BM25 retrieval + retrieval-only answers, no torch.
#   docker build -t pubmed-rag .
#   docker run -p 8000:8000 -v "$PWD/index:/app/index" pubmed-rag
#
# With the local LLM + dense search (needs a GPU host and a CUDA-enabled PyTorch base image):
#   docker build --build-arg BASE=<cuda-pytorch-image> --build-arg INSTALL_ML=1 -t pubmed-rag-gpu .
#   docker run --gpus all -e PMRAG_BACKEND=local -p 8000:8000 -v "$PWD/index:/app/index" pubmed-rag-gpu
ARG BASE=python:3.11-slim
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PMRAG_INDEX_DIR=/app/index \
    PMRAG_LOG_DIR=/app/logs

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_ML=0
RUN if [ "$INSTALL_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY src ./src
RUN useradd --create-home app && mkdir -p /app/index /app/logs && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "pubmed_rag.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

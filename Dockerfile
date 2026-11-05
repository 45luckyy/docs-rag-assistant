# Stage 1: fetch the corpus so the image is self-contained.
FROM python:3.13-slim AS corpus
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY scripts/fetch_corpus.sh .
RUN bash fetch_corpus.sh /src/corpus

# Stage 2: runtime.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    DOCS_DIR=/app/corpus INDEX_DIR=/app/data/index
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=corpus /src/corpus /app/corpus
COPY app ./app
COPY eval ./eval
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN useradd --create-home appuser && mkdir -p /app/data && chown -R appuser /app/data
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
ENTRYPOINT ["bash", "/app/entrypoint.sh"]

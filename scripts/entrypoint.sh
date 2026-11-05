#!/usr/bin/env bash
# Build the index on first start (persist /app/data as a volume to keep it), then serve.
set -euo pipefail

if [[ ! -f "${INDEX_DIR}/index.faiss" ]]; then
  echo "No index found, building it (provider=${PROVIDER:-offline})..."
  python -m app.ingest
fi
exec uvicorn app.api:app --host 0.0.0.0 --port 8000

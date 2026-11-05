#!/usr/bin/env bash
# Download the FastAPI docs (MIT licence) into data/corpus.
set -euo pipefail

DEST="${1:-data/corpus}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git "$TMP/fastapi"
git -C "$TMP/fastapi" sparse-checkout set docs/en/docs

SRC="$TMP/fastapi/docs/en/docs"
# Non-text assets and auto-generated API reference stubs add nothing to retrieval.
rm -rf "$SRC"/{img,css,js,reference}
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -r "$SRC" "$DEST"
echo "Corpus ready: $(find "$DEST" -name '*.md' | wc -l) markdown files in $DEST"

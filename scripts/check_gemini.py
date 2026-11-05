"""Quick check of the Gemini key and models in .env.

Usage: python scripts/check_gemini.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai  # noqa: E402

from app.config import get_settings  # noqa: E402

s = get_settings()
if not s.google_api_key:
    sys.exit("GOOGLE_API_KEY is empty in .env")
client = genai.Client(api_key=s.google_api_key)


def step(name, fn):
    t = time.time()
    try:
        out = fn()
        print(f"OK   {name} ({time.time() - t:.1f}s): {out}")
    except Exception as exc:
        print(f"FAIL {name} ({time.time() - t:.1f}s): {type(exc).__name__}: {str(exc)[:400]}")


step("list models", lambda: ", ".join(
    m.name.removeprefix("models/") for m in client.models.list()
    if "embed" in m.name or "flash" in m.name
))
def embed_dim():
    result = client.models.embed_content(model=s.embed_model, contents="hello")
    return f"dim={len(result.embeddings[0].values)}"


step(f"embed with {s.embed_model}", embed_dim)
step(f"chat with {s.llm_model}", lambda: client.models.generate_content(
    model=s.llm_model, contents="Reply with the word OK.").text.strip())

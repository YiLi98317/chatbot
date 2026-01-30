from qdrant_client import QdrantClient
import os
import sys
from pathlib import Path

# Ensure `src/` is on sys.path so we can import project modules when running directly.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot.config import get_settings
from chatbot.embeddings.ollama import embed_text
from chatbot.vectorstore.qdrant_store import QdrantStore

settings = get_settings()
COLL = os.getenv("QDRANT_COLLECTION", settings.default_collection)
TARGET_ID = "962ab6ec-5e08-5a79-a2d2-60a60f503d13"

doc_text = """Table: Track
TrackId: 1045
Name: Fly Me To The Moon
Composer: bart howard
"""

doc_text1 = "Fly Me To The Moon"

client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60.0)
store = QdrantStore(client, base_url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def embed(text: str) -> list[float]:
    return embed_text(text, model=settings.embed_model, base_url=settings.ollama_base_url)


qvec = embed(doc_text1)

hits = store.search(collection=COLL, query_vector=qvec, top_k=10)

print("Top 10 IDs:")
for i, h in enumerate(hits, 1):
    payload = getattr(h, "payload", {}) or {}
    hid = getattr(h, "id", None)
    try:
        score_val = float(getattr(h, "score", 0.0))
    except Exception:
        score_val = getattr(h, "score", 0.0)
    text_preview = (payload.get("text", "") or "").replace("\n", " ")
    if len(text_preview) > 120:
        text_preview = text_preview[:120] + "…"
    print(
        i, hid if hid is not None else "N/A", f"{score_val:.4f}", payload.get("table"), text_preview
    )

print("Target in top-10?", any(str(getattr(h, "id", "")) == TARGET_ID for h in hits))

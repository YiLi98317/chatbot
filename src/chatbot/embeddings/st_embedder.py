from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Optional

import unicodedata
import re

_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")


def _normalize_input(text: str) -> str:
    s = unicodedata.normalize("NFKC", text or "")
    s = _ZERO_WIDTH_RE.sub("", s)
    return s


@lru_cache(maxsize=2)
def _get_model(model_name: str):
    # Lazy import so environments without these deps can still run in Ollama mode.
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 32,
) -> List[List[float]]:
    if not texts:
        return []
    m = _get_model(model_name)
    normed = [_normalize_input(t) for t in texts]
    # sentence-transformers versions differ slightly; handle both.
    try:
        vecs = m.encode(
            normed,
            batch_size=int(batch_size),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
    except TypeError:
        vecs = m.encode(
            normed,
            batch_size=int(batch_size),
            show_progress_bar=False,
        )
    try:
        return vecs.tolist()  # type: ignore[return-value]
    except Exception:
        return [list(map(float, v)) for v in vecs]  # type: ignore[arg-type]


def embed_text(text: str, *, model_name: str) -> List[float]:
    out = embed_texts([text], model_name=model_name, batch_size=1)
    return out[0] if out else []


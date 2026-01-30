from __future__ import annotations

from typing import Iterable, List, Sequence

import requests


def _embeddings_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/embeddings"


def embed_text(text: str, model: str, base_url: str) -> List[float]:
    resp = requests.post(
        _embeddings_endpoint(base_url),
        json={"model": model, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError(f"Unexpected embeddings response: {data}")
    return embedding  # type: ignore[return-value]


def embed_texts(texts: Sequence[str], model: str, base_url: str) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for t in texts:
        embeddings.append(embed_text(t, model, base_url))
    return embeddings

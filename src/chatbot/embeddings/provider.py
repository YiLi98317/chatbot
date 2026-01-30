from __future__ import annotations

from typing import List, Sequence

from .ollama import embed_text as ollama_embed_text
from .ollama import embed_texts as ollama_embed_texts


def embed_texts(
    texts: Sequence[str],
    *,
    provider: str,
    model: str,
    ollama_base_url: str,
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Provider wrapper for embeddings.

    - provider=ollama: uses Ollama HTTP embeddings
    - provider=sentence_transformers: uses SentenceTransformers locally
    """
    p = (provider or "ollama").strip().lower()
    if p == "sentence_transformers":
        try:
            from .st_embedder import embed_texts as st_embed_texts
        except Exception as e:
            raise RuntimeError(
                "EMBED_PROVIDER=sentence_transformers but dependencies are not installed. "
                "Run: make install (or: .venv/bin/pip install -r requirements.txt). "
                f"Import error: {type(e).__name__}: {e}"
            ) from e

        return st_embed_texts(texts, model_name=model, batch_size=batch_size)
    return ollama_embed_texts(texts, model=model, base_url=ollama_base_url)


def embed_text(
    text: str,
    *,
    provider: str,
    model: str,
    ollama_base_url: str,
) -> List[float]:
    p = (provider or "ollama").strip().lower()
    if p == "sentence_transformers":
        try:
            from .st_embedder import embed_text as st_embed_text
        except Exception as e:
            raise RuntimeError(
                "EMBED_PROVIDER=sentence_transformers but dependencies are not installed. "
                "Run: make install (or: .venv/bin/pip install -r requirements.txt). "
                f"Import error: {type(e).__name__}: {e}"
            ) from e

        return st_embed_text(text, model_name=model)
    return ollama_embed_text(text, model=model, base_url=ollama_base_url)


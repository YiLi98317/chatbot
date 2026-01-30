from __future__ import annotations

import re
from typing import List


_ZH_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；…])")


def chunk_text_en(text: str, *, max_chars: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple fixed-size chunking with overlap for robust retrieval.
    """
    s = (text or "")
    if not s:
        return []
    max_chars = max(128, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))
    out: List[str] = []
    i = 0
    while i < len(s):
        j = min(len(s), i + max_chars)
        out.append(s[i:j])
        if j >= len(s):
            break
        i = max(0, j - overlap)
    return out


def chunk_text_zh(text: str, *, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Chinese-aware chunking:
    - Split on common Chinese sentence punctuation
    - Pack sentences into character-budget chunks
    - Character-based overlap
    """
    s = (text or "")
    if not s:
        return []
    max_chars = max(128, int(max_chars))
    overlap = max(0, min(int(overlap), max_chars // 2))

    parts = [p for p in _ZH_SENT_SPLIT_RE.split(s) if p]
    if not parts:
        return chunk_text_en(s, max_chars=max_chars, overlap=overlap)

    chunks: List[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # If a single part is too large, fall back to fixed-size slicing for that part.
        if len(part) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(chunk_text_en(part, max_chars=max_chars, overlap=overlap))
            continue

        if not buf:
            buf = part
            continue

        if len(buf) + len(part) <= max_chars:
            buf += part
            continue

        # Flush current buffer and start a new chunk with overlap.
        chunks.append(buf)
        tail = buf[-overlap:] if overlap > 0 else ""
        buf = (tail + part) if tail else part

    if buf:
        chunks.append(buf)

    return chunks


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 120) -> List[str]:
    """
    Backward-compatible alias for English chunking.
    """
    return chunk_text_en(text, max_chars=max_chars, overlap=overlap)


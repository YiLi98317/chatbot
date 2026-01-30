from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def content_hash(s: str) -> str:
    return hashlib.sha256(_norm(s).encode("utf-8")).hexdigest()


def deduplicate_texts(texts: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for t in texts:
        h = content_hash(t)
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class SubQuery:
    text: str
    purpose: str  # e.g., "find_entity", "fetch_attribute"


def decompose_query(query: str) -> List[SubQuery]:
    """
    Naive deterministic decomposition:
      - if pattern 'X from album Y' -> ['album Y', 'tracks of album Y']
      - else no-op (single-hop)
    """
    s = (query or "").strip().lower()
    if "tracks from album" in s or "tracks of album" in s:
        # extract quoted album name if any
        import re
        m = re.search(r'"([^"]+)"', query) or re.search(r"'([^']+)'", query)
        if m:
            album = m.group(1).strip()
            return [
                SubQuery(text=f'album "{album}"', purpose="find_entity"),
                SubQuery(text=f'tracks from album "{album}"', purpose="fetch_attribute"),
            ]
    return [SubQuery(text=query, purpose="single_hop")]


def stitch_summaries(snippets: List[str]) -> str:
    """
    Simple concatenation with separators; caller should enforce token budgets upstream.
    """
    return "\n---\n".join(snippets)


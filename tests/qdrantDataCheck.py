#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import sys
from pathlib import Path

# Ensure `src/` is on sys.path so `import chatbot` works when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot.config import get_settings


def exists_in_qdrant(
    needle: str,
    collection: Optional[str] = None,
    table: str = "Track",
    page_size: int = 256,
    case_insensitive: bool = True,
) -> Tuple[bool, Optional[object]]:
    """
    Scrolls Qdrant with a payload filter on `table` and checks if `needle`
    appears in the payload `text` of any point.
    Returns (found, point) where point is the first matching point (or None).
    """
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60.0)
    collection_name = collection or settings.default_collection

    # Ensure payload index for 'table' so we can filter server-side
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="table",
            field_schema="keyword",  # compatible across client versions
        )
    except Exception:
        # Ignore if it already exists or server is older; scroll will still work without filter if needed
        pass

    # Filter to only scan points ingested from the desired table
    scroll_filter = Filter(must=[FieldCondition(key="table", match=MatchValue(value=table))])

    target = needle.lower() if case_insensitive else needle
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=page_size,
            offset=next_offset,
        )

        for pt in points:
            payload = getattr(pt, "payload", None) or {}
            text = payload.get("text", "")
            haystack = text.lower() if case_insensitive else text
            if target in haystack:
                return True, pt  # Found

        if next_offset is None:
            break

    return False, None


if __name__ == "__main__":
    found, point = exists_in_qdrant("Fly Me To The Moon", table="Track")
    print(f"Exists: {found}")
    if found and point is not None:
        print(f"Point ID: {point.id}")
        print(f"Source: {point.payload.get('source')}")
        print(f"Snippet:\n{point.payload.get('text', '')[:400]}")

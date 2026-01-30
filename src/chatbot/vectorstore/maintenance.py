from __future__ import annotations

from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import PayloadSchemaType


def ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name="table",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


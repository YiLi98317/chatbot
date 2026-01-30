from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from types import SimpleNamespace

import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, SearchRequest, PayloadSchemaType
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException


class QdrantStore:
    def __init__(
        self, client: QdrantClient, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        self.client = client
        self._base_url = base_url
        self._api_key = api_key

    def _resolve_points_api(self):
        # Prefer the same path used by upsert in this client: _client.openapi_client.points_api
        candidate_paths = [
            ("_client", "openapi_client", "points_api"),
            ("openapi_client", "points_api"),
            ("http", "api", "points_api"),
            ("http", "points_api"),
        ]
        for path in candidate_paths:
            obj = self.client
            ok = True
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    ok = False
                    break
            if ok and hasattr(obj, "search_points"):
                return obj
        return None

    def ensure_collection(
        self, name: str, vector_size: int, distance: Distance = Distance.COSINE
    ) -> None:
        """
        Ensure collection exists with expected vector size. If it exists but size differs, recreate it.
        """
        try:
            # Check existence
            exists = False
            if hasattr(self.client, "collection_exists"):
                try:
                    exists = bool(self.client.collection_exists(collection_name=name))  # type: ignore[attr-defined]
                except Exception:
                    exists = False
            else:
                try:
                    self.client.get_collection(name)
                    exists = True
                except Exception:
                    exists = False

            if not exists:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(size=vector_size, distance=distance),
                )
                # Ensure payload indexes for filtering
                self.ensure_default_payload_indexes(name)
                return

            # If exists, compare vector size; if mismatch, delete and create
            try:
                info = self.client.get_collection(name)
                current = info.config.params.vectors  # type: ignore[attr-defined]
                if isinstance(current, dict):
                    current_size = current.get("size")
                    current_distance = current.get("distance")
                else:
                    current_size = getattr(current, "size", None)
                    current_distance = getattr(current, "distance", None)
                if current_size != vector_size or (
                    current_distance and current_distance != distance
                ):
                    self.client.delete_collection(collection_name=name)
                    self.client.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(size=vector_size, distance=distance),
                    )
                # Ensure payload indexes either way
                self.ensure_default_payload_indexes(name)
            except Exception:
                # If inspection fails, assume OK and proceed
                try:
                    self.ensure_default_payload_indexes(name)
                except Exception:
                    pass
        except Exception:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            try:
                self.ensure_default_payload_indexes(name)
            except Exception:
                pass

    def ensure_default_payload_indexes(self, name: str) -> None:
        """
        Ensure common payload indexes exist to support filters used by this app.
        Currently ensures:
        - 'table': KEYWORD
        - 'lang': KEYWORD
        """
        try:
            self.client.create_payload_index(
                collection_name=name,
                field_name="table",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            # Already exists or server doesn't support schema endpoint
            pass
        try:
            self.client.create_payload_index(
                collection_name=name,
                field_name="lang",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def upsert_texts(
        self,
        collection: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 128,
    ) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("metadatas must have the same length as texts")

        def gen_points() -> Iterable[PointStruct]:
            for i, (text, vector) in enumerate(zip(texts, embeddings)):
                payload = {"text": text}
                if metadatas:
                    payload.update(metadatas[i])
                point_id = ids[i] if ids else str(uuid.uuid4())
                yield PointStruct(id=point_id, vector=vector, payload=payload)

        buffer: List[PointStruct] = []

        def _try_upsert(points: List[PointStruct]) -> None:
            self.client.upsert(collection_name=collection, points=points)

        def flush(points: List[PointStruct]) -> None:
            if not points:
                return
            try:
                _try_upsert(points)
            except UnexpectedResponse:
                # Likely vector size mismatch; recreate with inferred size and retry once (without deprecated API)
                inferred_size = (
                    len(points[0].vector) if points and points[0].vector is not None else None
                )
                if inferred_size:
                    try:
                        if hasattr(
                            self.client, "collection_exists"
                        ) and self.client.collection_exists(collection_name=collection):  # type: ignore[attr-defined]
                            self.client.delete_collection(collection_name=collection)
                    except Exception:
                        # best-effort delete
                        pass
                    self.client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=inferred_size, distance=Distance.COSINE),
                    )
                    _try_upsert(points)
                else:
                    raise
            except Exception:
                # Network/protocol issues (e.g., Broken pipe). Split batch and retry recursively.
                if len(points) <= 1:
                    raise
                mid = len(points) // 2
                left = points[:mid]
                right = points[mid:]
                flush(left)
                flush(right)

        for p in gen_points():
            buffer.append(p)
            if len(buffer) >= batch_size:
                flush(buffer)
                buffer.clear()
        if buffer:
            flush(buffer)

    def search(
        self,
        collection: str,
        query_vector: Sequence[float],
        top_k: int = 4,
        filters: Optional[Filter] = None,
        debug: bool = False,
    ):
        # 1) Try high-level search API first
        if hasattr(self.client, "search"):
            try:
                return self.client.search(
                    collection_name=collection,
                    query_vector=list(query_vector),
                    query_filter=filters,
                    limit=top_k,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                if debug:
                    try:
                        print("QDRANT_SEARCH_ERROR_HIGHLEVEL:", type(e).__name__, str(e))
                    except Exception:
                        pass
                pass
        # 2) Try high-level query_points (maps to /points/query) if available
        # Intentionally omitted to avoid version-dependent API differences.
        # 3) Fallback: direct REST call to /collections/{collection}/points/search
        if not self._base_url:
            raise RuntimeError(
                "qdrant-client lacks search/query_points and no base_url provided for HTTP fallback."
            )
        url = f"{self._base_url.rstrip('/')}/collections/{collection}/points/search"
        body: Dict[str, Any] = {
            "vector": list(query_vector),
            "limit": int(top_k),
            "with_payload": True,
        }
        if filters is not None:
            try:
                body["filter"] = filters.dict(by_alias=True)
            except Exception:
                pass
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["api-key"] = self._api_key
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError as he:
            if debug:
                try:
                    status = resp.status_code
                    resp_text = ""
                    try:
                        resp_text = resp.text[:2000]
                    except Exception:
                        pass
                    vec_len = len(query_vector) if hasattr(query_vector, "__len__") else None
                    vec_head = list(query_vector[:8]) if hasattr(query_vector, "__getitem__") else []
                    current_size = None
                    try:
                        info = self.client.get_collection(collection)
                        vectors_cfg = getattr(info.config.params, "vectors", None)  # type: ignore[attr-defined]
                        if isinstance(vectors_cfg, dict):
                            current_size = vectors_cfg.get("size")
                        else:
                            current_size = getattr(vectors_cfg, "size", None)
                    except Exception:
                        current_size = None
                    print("QDRANT_400_DEBUG:")
                    print("  url:", url)
                    print("  status:", status)
                    print("  req.limit:", body.get("limit"))
                    print("  req.vector_len:", vec_len)
                    print("  req.vector_head:", vec_head)
                    print("  req.has_filter:", "filter" in body)
                    print("  collection_vector_size:", current_size)
                    if current_size and vec_len and current_size != vec_len:
                        print(f"  MISMATCH: collection size {current_size} vs embed size {vec_len}")
                    print("  resp.body.head:", resp_text[:400])
                except Exception:
                    pass
            raise
        data = resp.json()
        result = data.get("result")
        # /points/search returns a list of points directly
        points = result if isinstance(result, list) else (result.get("points") if isinstance(result, dict) else [])
        points = points or []
        return [
            SimpleNamespace(id=p.get("id"), payload=p.get("payload"), score=p.get("score"))
            for p in points
        ]

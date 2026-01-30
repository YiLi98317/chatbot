#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure `src/` is on sys.path so `import chatbot` works when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dotenv import load_dotenv  # type: ignore  # noqa: E402
from chatbot.config import get_settings  # type: ignore  # noqa: E402
from chatbot.cli.ingest_sql import ingest_sql  # type: ignore  # noqa: E402
from chatbot.embeddings.ollama import embed_text  # type: ignore  # noqa: E402
from qdrant_client import QdrantClient  # type: ignore  # noqa: E402


def build_mysql_db_uri() -> str:
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DB")

    missing = [k for k, v in {
        "MYSQL_USER": user,
        "MYSQL_PASSWORD": password,
        "MYSQL_DB": database
    }.items() if not v]
    if missing:
        raise RuntimeError(
            f"Missing required MySQL env vars: {', '.join(missing)}. "
            "Set them in your environment or .env."
        )

    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"


def main() -> None:
    # Load .env before reading env vars
    load_dotenv()
    # Prefer DB_URI if provided, else build from MYSQL_* env vars
    db_uri = os.getenv("DB_URI") or build_mysql_db_uri()
    os.environ["DB_URI"] = db_uri

    # Load settings (respects .env and the DB_URI we just set)
    settings = get_settings()

    collection_name = os.getenv("QDRANT_COLLECTION", settings.default_collection)
    embed_model = os.getenv("EMBED_MODEL", getattr(settings, "embed_model", "nomic-embed-text"))

    # Determine embedding dimension to compare with existing Qdrant collection
    target_vector_size = None
    try:
        sample_emb = embed_text("dimension_check", model=embed_model, base_url=settings.ollama_base_url)
        target_vector_size = len(sample_emb)
    except Exception:
        # If Ollama isn't up yet, we won't be able to preflight vector size; proceed with default behavior.
        pass

    # Decide whether to force recreate based on collection vector size mismatch
    force_recreate = False
    if target_vector_size is not None:
        try:
            qc = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60.0)
            info = qc.get_collection(collection_name)
            # Qdrant client may expose vectors config as dict or object; handle both.
            current_size = None
            try:
                # newer clients
                current_size = info.config.params.vectors.size  # type: ignore[attr-defined]
            except Exception:
                try:
                    current_size = info.config.params.vectors["size"]  # type: ignore[index]
                except Exception:
                    current_size = None
            if current_size is not None and current_size != target_vector_size:
                print(
                    f"Qdrant collection '{collection_name}' vector size {current_size} != embedding size {target_vector_size}; "
                    f"will recreate collection."
                )
                force_recreate = True
        except Exception:
            # Collection not found or client error → let first table create it
            pass

    # Chinook tables and columns (use PK as updated-at surrogate for deterministic ordering)
    # For PlaylistTrack (composite PK), use both IDs split across pk/updated_at to ensure unique doc IDs.
    tables: List[Tuple[str, str, str]] = [
        ("Artist", "ArtistId", "ArtistId"),
        ("Album", "AlbumId", "AlbumId"),
        ("Track", "TrackId", "TrackId"),
        ("Genre", "GenreId", "GenreId"),
        ("MediaType", "MediaTypeId", "MediaTypeId"),
        ("Playlist", "PlaylistId", "PlaylistId"),
        ("PlaylistTrack", "PlaylistId", "TrackId"),  # composite uniqueness via both columns
        ("Customer", "CustomerId", "CustomerId"),
        ("Invoice", "InvoiceId", "InvoiceId"),
        ("InvoiceLine", "InvoiceLineId", "InvoiceLineId"),
        ("Employee", "EmployeeId", "EmployeeId"),
    ]

    # Batch size (smaller batches reduce likelihood of HTTP broken pipe)
    batch_size_env = os.getenv("QDRANT_BATCH_SIZE")
    try:
        batch_size = int(batch_size_env) if batch_size_env else 32
    except ValueError:
        batch_size = 32

    print(f"Re-ingesting Chinook MySQL DB: {db_uri}")
    print(f"Collection: {collection_name} | Embed model: {embed_model}")
    print(f"Batch size: {batch_size}")

    # First table recreates the collection; the rest append
    for idx, (table, pk_col, updated_at_col) in enumerate(tables):
        recreate = (idx == 0) or force_recreate
        print(
            f"[{idx + 1}/{len(tables)}] Ingesting table={table} pk={pk_col} updated_at={updated_at_col} "
            f"{'(recreate)' if recreate else ''}"
        )
        ingest_sql(
            table=table,
            since=None,
            where=None,
            limit=None,
            collection=collection_name,
            embed_model=embed_model,
            recreate=recreate,
            pk_col=pk_col,
            updated_at_col=updated_at_col,
            batch_size=batch_size,
        )
    print("Done.")


if __name__ == "__main__":
    main()


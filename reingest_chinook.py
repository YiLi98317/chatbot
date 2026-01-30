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

from chatbot.config import get_settings  # type: ignore  # noqa: E402
from chatbot.cli.ingest_sql import ingest_sql  # type: ignore  # noqa: E402


def main() -> None:
    # Default DB path to Chinook sqlite if DB_URI not provided
    default_db_path = PROJECT_ROOT / "data" / "chinook" / "Chinook_Sqlite.sqlite"
    db_uri = os.getenv("DB_URI") or f"sqlite:///{default_db_path}"
    os.environ["DB_URI"] = db_uri

    # Load settings (respects .env and the DB_URI we just set)
    settings = get_settings()

    collection_name = os.getenv("QDRANT_COLLECTION", settings.default_collection)
    embed_model = os.getenv("EMBED_MODEL", getattr(settings, "embed_model", "nomic-embed-text"))

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

    print(f"Re-ingesting Chinook DB: {db_uri}")
    print(f"Collection: {collection_name} | Embed model: {embed_model}")
    print(f"Batch size: {batch_size}")

    # First table recreates the collection; the rest append
    for idx, (table, pk_col, updated_at_col) in enumerate(tables):
        recreate = idx == 0
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

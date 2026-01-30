from __future__ import annotations

from typing import Iterable, Optional
from sqlalchemy import text
from sqlalchemy.engine import Engine, RowMapping


def select_rows(
    engine: Engine,
    table: str,
    pk_col: str,
    updated_at_col: str = "updated_at",
    since: Optional[str] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterable[RowMapping]:
    """
    Yields rows as mappings from a table with optional since/where filters.
    WARNING: 'where' is injected as raw SQL clause. Prefer using vetted values.
    """
    clauses = []
    params = {}
    if since:
        clauses.append(f"{updated_at_col} >= :since")
        params["since"] = since
    if where:
        clauses.append(f"({where})")
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    sql = text(
        f"""
        SELECT * FROM {table}
        {where_sql}
        ORDER BY {updated_at_col} ASC, {pk_col} ASC
        {limit_sql}
        """
    )
    with engine.begin() as conn:
        result = conn.execute(sql, params)
        for row in result.mappings():
            yield row


def get_schema_graph() -> dict:
    """
    Return a simple PK/FK schema graph for Chinook core tables.
    This is a static map sufficient for basic join validation and planning.
    """
    return {
        "Track": {
            "pk": "TrackId",
            "fks": {"AlbumId": ("Album", "AlbumId"), "GenreId": ("Genre", "GenreId"), "MediaTypeId": ("MediaType", "MediaTypeId")},
        },
        "Album": {
            "pk": "AlbumId",
            "fks": {"ArtistId": ("Artist", "ArtistId")},
        },
        "Artist": {"pk": "ArtistId", "fks": {}},
        "Genre": {"pk": "GenreId", "fks": {}},
        "MediaType": {"pk": "MediaTypeId", "fks": {}},
        "Playlist": {"pk": "PlaylistId", "fks": {}},
        "PlaylistTrack": {
            "pk": "PlaylistId:TrackId",
            "fks": {"PlaylistId": ("Playlist", "PlaylistId"), "TrackId": ("Track", "TrackId")},
        },
        "Customer": {"pk": "CustomerId", "fks": {}},
        "Employee": {"pk": "EmployeeId", "fks": {}},
        "Invoice": {"pk": "InvoiceId", "fks": {"CustomerId": ("Customer", "CustomerId")}},
        "InvoiceLine": {
            "pk": "InvoiceLineId",
            "fks": {"InvoiceId": ("Invoice", "InvoiceId"), "TrackId": ("Track", "TrackId")},
        },
    }

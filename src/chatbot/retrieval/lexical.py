from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Row
from chatbot.retrieval.normalize import detect_lang, normalize_text, tokenize_en


@dataclass(frozen=True)
class Hit:
    table: str
    pk: int
    title_field: str
    title_value: str
    confidence: float
    row: Dict


WRAPPER_PATTERNS = [
    r"^\s*give\s+me\s+info\s+about\s+",
    r"^\s*tell\s+me\s+about\s+",
    r"^\s*info\s+about\s+",
    r"^\s*name\s*:\s*",
]


def normalize_query(query: str) -> str:
    """
    Strip common wrappers like:
      - 'give me info about ...'
      - 'tell me about ...'
      - 'info about ...'
      - 'name: ...'
    Also trims quotes and surrounding whitespace.
    """
    s = query or ""
    s_strip = s.strip()
    lower = s_strip.lower()
    for pat in WRAPPER_PATTERNS:
        m = re.match(pat, lower)
        if m:
            # remove the matched prefix from the original string by length
            s_strip = s_strip[m.end() :]
            break
    # remove surrounding quotes if present
    s_strip = s_strip.strip().strip("'").strip('"').strip()
    return s_strip


def tokenize(s: str) -> List[str]:
    """
    Tokenization for lexical SQL lookup.

    - en: split into ASCII-ish tokens (backward compatible)
    - zh: return the whole normalized string as a single token (so LIKE %token% works)
    - mixed: prefer the English tokens (keeps behavior predictable for this CLI helper)
    """
    lang = detect_lang(s)
    if lang == "en":
        return tokenize_en(s)
    if lang == "zh":
        q = normalize_text(s, lang="zh").strip()
        return [q] if q else []
    # mixed
    return tokenize_en(s)


def _ensure_engine(db_uri: str) -> Engine:
    return create_engine(db_uri)


def _exact_match_sql(table: str, field: str) -> str:
    return f"SELECT * FROM {table} WHERE lower({field}) = :name LIMIT :limit"


def _token_and_sql(table: str, field: str, num_tokens: int) -> str:
    clauses = [f"lower({field}) LIKE :tok{i}" for i in range(num_tokens)]
    where_sql = " AND ".join(clauses) if clauses else "1=1"
    return f"SELECT * FROM {table} WHERE {where_sql} LIMIT :limit"


def _pk_field_for_table(table: str) -> str:
    if table == "Track":
        return "TrackId"
    if table == "Album":
        return "AlbumId"
    if table == "Artist":
        return "ArtistId"
    return "id"


def _title_field_for_table(table: str) -> str:
    if table == "Track":
        return "Name"
    if table == "Album":
        return "Title"
    if table == "Artist":
        return "Name"
    return "name"


def _row_to_mapping(row: Row) -> Dict:
    try:
        # RowMapping-like behavior via _mapping
        return dict(row._mapping)  # type: ignore[attr-defined]
    except Exception:
        try:
            return dict(row)  # type: ignore[arg-type]
        except Exception:
            # Best effort; nothing better to do
            return {}


def lexical_lookup(
    db_uri: str,
    candidate: str,
    preferred_tables: List[str],
    limit: int = 5,
) -> List[Hit]:
    """
    Perform lexical lookup in Chinook DB:
      1) exact case-insensitive match on key fields
      2) token-AND LIKE match on same fields
    Returns up to `limit` hits total in preferred table order.
    """
    if not candidate or not candidate.strip():
        return []
    total_limit = max(1, int(limit))
    engine = _ensure_engine(db_uri)
    lower_name = candidate.strip().lower()
    tokens = [t.lower() for t in tokenize(candidate)]

    hits: List[Hit] = []

    def run_exact(table: str) -> Iterable[Hit]:
        field = _title_field_for_table(table)
        sql = _exact_match_sql(table, field)
        params = {"name": lower_name, "limit": total_limit}
        with engine.begin() as conn:
            for row in conn.execute(text(sql), params):
                m = _row_to_mapping(row)
                pk_field = _pk_field_for_table(table)
                pk_val = int(m.get(pk_field))
                title_val = str(m.get(field) or "")
                yield Hit(
                    table=table,
                    pk=pk_val,
                    title_field=field,
                    title_value=title_val,
                    confidence=1.0,
                    row=m,
                )

    def run_token_and(table: str) -> Iterable[Hit]:
        if not tokens:
            return []
        field = _title_field_for_table(table)
        sql = _token_and_sql(table, field, len(tokens))
        params = {f"tok{i}": f"%{tok}%" for i, tok in enumerate(tokens)}
        params["limit"] = total_limit
        with engine.begin() as conn:
            for row in conn.execute(text(sql), params):
                m = _row_to_mapping(row)
                pk_field = _pk_field_for_table(table)
                pk_val = int(m.get(pk_field))
                title_val = str(m.get(field) or "")
                yield Hit(
                    table=table,
                    pk=pk_val,
                    title_field=field,
                    title_value=title_val,
                    confidence=0.8,
                    row=m,
                )

    # 1) exact across preferred tables
    for table in preferred_tables:
        for h in run_exact(table):
            hits.append(h)
            if len(hits) >= total_limit:
                return hits

    # 2) token-AND across preferred tables
    for table in preferred_tables:
        for h in run_token_and(table):
            hits.append(h)
            if len(hits) >= total_limit:
                return hits

    return hits



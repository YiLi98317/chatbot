from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Reuse the existing FTS index/query utilities
from .entity_resolver import ensure_fts_index, fts_search, infer_entity_types
import math
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Row
from chatbot.retrieval.lexical import tokenize as lexical_tokenize
from chatbot.retrieval.lexical import _title_field_for_table as title_field_for_table  # type: ignore
from chatbot.retrieval.lexical import _pk_field_for_table as pk_field_for_table  # type: ignore
from chatbot.sql.row_to_doc import row_to_text


@dataclass(frozen=True)
class BM25Hit:
    entity_type: str
    entity_id: int
    name: str
    extra: str
    bm25: float
    score: float  # normalized to [0, 1], higher is better


def _normalize_bm25(rows: List[Dict]) -> List[BM25Hit]:
    """
    Normalize SQLite FTS5 bm25 (lower-is-better) to [0, 1] (higher-is-better).
    score = (bmax - bm25) / (bmax - bmin + 1e-6)
    """
    if not rows:
        return []
    vals = [float(m.get("rank", 0.0)) for m in rows]
    bmin = min(vals)
    bmax = max(vals)
    denom = (bmax - bmin) if (bmax - bmin) > 1e-6 else 1.0
    out: List[BM25Hit] = []
    for m in rows:
        et = str(m.get("entity_type", ""))
        eid = int(m.get("entity_id", 0))
        name = str(m.get("name", "") or "")
        extra = str(m.get("extra", "") or "")
        bm25_raw = float(m.get("rank", 0.0))
        score = (bmax - bm25_raw) / denom
        out.append(BM25Hit(entity_type=et, entity_id=eid, name=name, extra=extra, bm25=bm25_raw, score=score))
    return out


def search_bm25(
    db_uri: str,
    query: str,
    *,
    entity_types: Optional[Sequence[str]] = None,
    limit: int = 50,
) -> List[BM25Hit]:
    """
    Run BM25 retrieval against the existing FTS index and return hits with scores in [0, 1].
    """
    ensure_fts_index(db_uri)
    types = list(entity_types) if entity_types else infer_entity_types(query)
    raw_rows = fts_search(db_uri=db_uri, candidate=query, entity_types=types, limit=limit)
    return _normalize_bm25(raw_rows)


def search_bm25_as_dicts(
    db_uri: str,
    query: str,
    *,
    entity_types: Optional[Sequence[str]] = None,
    limit: int = 50,
) -> List[Dict]:
    """
    Convenience wrapper returning plain dicts suitable for downstream merging:
      - score: normalized [0, 1]
      - metadata: includes bm25_raw and provenance
    """
    hits = search_bm25(db_uri=db_uri, query=query, entity_types=entity_types, limit=limit)
    out: List[Dict] = []
    for h in hits:
        out.append(
            {
                "text": h.name,
                "score": float(h.score),
                "metadata": {
                    "source": "bm25",
                    "bm25_raw": float(h.bm25),
                    "entity_type": h.entity_type,
                    "entity_id": h.entity_id,
                    "name": h.name,
                    "extra": h.extra,
                },
            }
        )
    return out
 
@dataclass(frozen=True)
class _Doc:
    table: str
    pk: int
    title_value: str
    row: Dict


def _ensure_engine(db_uri: str) -> Engine:
    return create_engine(db_uri)


def _fetch_token_and_candidates(
    engine: Engine,
    table: str,
    field: str,
    terms: Sequence[str],
    limit: int,
) -> Iterable[_Doc]:
    """
    Fetch candidates using token-AND LIKE to keep compatibility when FTS is unavailable.
    """
    if not terms:
        return []
    clauses = [f"lower({field}) LIKE :tok{i}" for i, _ in enumerate(terms)]
    where_sql = " AND ".join(clauses)
    sql = text(f"SELECT * FROM {table} WHERE {where_sql} LIMIT :limit")
    params: Dict[str, object] = {f"tok{i}": f"%{t.lower()}%" for i, t in enumerate(terms)}
    params["limit"] = int(max(1, limit))
    pk_field = pk_field_for_table(table)
    with engine.begin() as conn:
        for row in conn.execute(sql, params):
            m = dict(row._mapping)  # type: ignore[attr-defined]
            pk_val = int(m.get(pk_field))
            title_val = str(m.get(field) or "")
            yield _Doc(table=table, pk=pk_val, title_value=title_val, row=m)


def _corpus_stats(engine: Engine, table: str, field: str, sample_for_avg: int = 500) -> Tuple[int, float]:
    """
    Return (N, avgdl_tokens). N is total rows in table.
    avgdl_tokens is approximated by sampling title lengths (chars) and converting to tokens (~5 chars/token).
    """
    with engine.begin() as conn:
        N = conn.execute(text(f"SELECT COUNT(*) AS n FROM {table}")).scalar_one()
        # Compute average char length over a sample to avoid scanning entire table
        sample_sql = text(
            f"SELECT AVG(LENGTH({field})) AS avglen FROM {table} "
            f"WHERE {field} IS NOT NULL LIMIT :lim"
        )
        avg_chars = conn.execute(sample_sql, {"lim": int(max(1, sample_for_avg))}).scalar_one()
    try:
        avg_chars_f = float(avg_chars or 0.0)
    except Exception:
        avg_chars_f = 0.0
    # Heuristic: ~5 chars per token on average (English, rough)
    avgdl = max(1.0, avg_chars_f / 5.0)
    return int(N or 0), float(avgdl)


def _doc_freq(engine: Engine, table: str, field: str, term: str) -> int:
    with engine.begin() as conn:
        sql = text(f"SELECT COUNT(*) AS n FROM {table} WHERE lower({field}) LIKE :t")
        n = conn.execute(sql, {"t": f"%{term.lower()}%"}).scalar_one()
    return int(n or 0)


def _bm25_score(
    title_tokens: Sequence[str],
    query_terms: Sequence[str],
    N: int,
    avgdl: float,
    k1: float = 1.2,
    b: float = 0.75,
    df_lookup: Optional[Dict[str, int]] = None,
) -> float:
    if not title_tokens or not query_terms or N <= 0:
        return 0.0
    dl = float(len(title_tokens))
    tf_counts: Dict[str, int] = {}
    for t in title_tokens:
        tf_counts[t] = tf_counts.get(t, 0) + 1
    score = 0.0
    for q in query_terms:
        tf = float(tf_counts.get(q, 0))
        if tf <= 0.0:
            continue
        n_q = 0
        if df_lookup is not None and q in df_lookup:
            n_q = int(df_lookup[q])
        # Guard against degenerate df lookup; if 0, treat as rare
        n_q = max(1, n_q)
        # Okapi BM25 idf
        idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
        numer = tf * (k1 + 1.0)
        denom = tf + k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
        score += idf * (numer / max(1e-9, denom))
    return float(score)


def _min_max_normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax <= vmin:
        # Degenerate: all equal -> 1.0
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def bm25_search(
    db_uri: str,
    query: str,
    preferred_tables: List[str],
    limit: int = 10,
    debug: bool = False,
) -> List[Dict]:
    """
    BM25-like retrieval over SQLite tables without requiring FTS:
    - Tokenize the query (no stopword removal to preserve recall)
    - For each preferred table, fetch candidates via token-AND LIKE
    - Compute BM25 scores using LIKE-based doc frequency as heuristic df
    - Normalize scores to [0,1] across all candidates
    - Return row_to_text payloads with provenance metadata
    """
    if not query or not query.strip() or not preferred_tables:
        return []
    engine = _ensure_engine(db_uri)
    terms = [t.lower() for t in lexical_tokenize(query) if t.strip()]
    if not terms:
        return []

    # Collect candidates across tables (overfetch to improve quality before cut)
    overfetch = max(limit * 3, 15)
    candidates: List[Tuple[_Doc, float]] = []

    for table in preferred_tables:
        field = title_field_for_table(table)
        try:
            N, avgdl = _corpus_stats(engine, table, field)
        except Exception:
            # Fall back if stats query fails
            N, avgdl = (1000, 8.0)
        # Pre-compute df per term for this table
        df_lookup: Dict[str, int] = {}
        for t in terms:
            try:
                df_lookup[t] = _doc_freq(engine, table, field, t)
            except Exception:
                df_lookup[t] = 1

        # Fetch docs and score
        fetched = list(_fetch_token_and_candidates(engine, table, field, terms, overfetch))
        for doc in fetched:
            title_tokens = [x.lower() for x in lexical_tokenize(doc.title_value)]
            score = _bm25_score(title_tokens, terms, N=N, avgdl=avgdl, df_lookup=df_lookup)
            candidates.append((doc, score))

    if not candidates:
        return []

    # Sort, take top-k, normalize
    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[: max(1, limit)]
    raw_scores = [s for _, s in top]
    norm_scores = _min_max_normalize(raw_scores)

    results: List[Dict] = []
    for (doc, _), ns in zip(top, norm_scores):
        # Build display text and metadata
        text_val = row_to_text(doc.table, doc.row)
        metadata: Dict[str, object] = {
            "table": doc.table,
            "pk": doc.pk,
            "source": f"bm25:{doc.table}:{doc.pk}",
        }
        # Include common helpful fields when present
        if doc.table == "Track":
            for k in ("TrackId", "Name", "Composer", "AlbumId", "GenreId"):
                if k in doc.row:
                    metadata[k] = doc.row[k]
        elif doc.table == "Album":
            for k in ("AlbumId", "Title", "ArtistId"):
                if k in doc.row:
                    metadata[k] = doc.row[k]
        elif doc.table == "Artist":
            for k in ("ArtistId", "Name"):
                if k in doc.row:
                    metadata[k] = doc.row[k]
        results.append(
            {
                "text": text_val,
                "score": float(ns),
                "metadata": metadata,
            }
        )

    return results


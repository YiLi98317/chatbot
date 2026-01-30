from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from rapidfuzz import fuzz
try:
    # Use centralized normalization when available
    from .normalize import normalize_for_fuzzy, detect_lang, normalize_text
except Exception:
    normalize_for_fuzzy = None  # type: ignore[assignment]
    detect_lang = None  # type: ignore[assignment]
    normalize_text = None  # type: ignore[assignment]


# -------------------------
# Data structures
# -------------------------

SUPPORTED_ENTITY_TYPES = {
    "Track",
    "Album",
    "Artist",
    "Genre",
    "MediaType",
    "Playlist",
    "PlaylistTrack",
    "Customer",
    "Employee",
    "Invoice",
    "InvoiceLine",
}

@dataclass(frozen=True)
class FTSCandidate:
    entity_type: str  # Track | Album | Artist
    entity_id: int
    name: str
    extra: str
    bm25: float
    bm25_norm: float
    fuzzy: float
    final: float


# -------------------------
# Candidate extraction
# -------------------------

_PUNCT_RE = re.compile(r"[^\w\s']+", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Z]+")

# Shared stop words for FTS token building and coverage checks
STOP_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "about",
    "named",
    "called",
    "titled",
    "song",
    "track",
    "album",
    "artist",
    "do",
    "you",
    "know",
    "is",
    "there",
    "me",
    "my",
    "your",
    "and",
    "or",
}

def _normalize_simple(s: str) -> str:
    """
    Lowercase, remove punctuation, collapse whitespace.
    Keep apostrophes to not break contractions; FTS and fuzzy are robust.
    """
    if not s:
        return ""
    s2 = s.lower().strip()
    s2 = _PUNCT_RE.sub(" ", s2)
    s2 = _SPACE_RE.sub(" ", s2).strip()
    return s2


def extract_candidate_strings(query: str) -> List[str]:
    """
    Extract normalized candidate strings from a natural-language query.
    - Prefer quoted phrases if present.
    - Otherwise, use simple tokenize + stopword removal (no wrapper regex).
    Returns [primary].
    """
    s = (query or "").strip()
    if not s:
        return []

    # Prefer first quoted phrase (double or single)
    m = re.search(r'"([^"]{2,})"', s)
    if m:
        primary = m.group(1).strip()
        return [primary]
    m = re.search(r"'([^']{2,})'", s)
    if m:
        primary = m.group(1).strip()
        return [primary]

    # If query contains CJK, do not apply English tokenization/stopwords here.
    try:
        lang = detect_lang(s) if detect_lang is not None else "en"
    except Exception:
        lang = "en"
    if lang in ("zh", "mixed"):
        try:
            norm = normalize_text(s, lang=lang) if normalize_text is not None else s
        except Exception:
            norm = s
        norm = (norm or "").strip()
        return [norm] if norm else []

    # Tokenize and remove lightweight stopwords; cap to 12 tokens to keep it focused
    tokens = [t for t in _TOKEN_SPLIT_RE.split(s) if t]
    if not tokens:
        return []
    filtered = [t for t in tokens if t.lower() not in STOP_WORDS]
    cand_tokens = filtered if filtered else tokens
    candidate = " ".join(cand_tokens[:12]).strip()
    return [candidate] if candidate else []


def infer_entity_types(query: str) -> List[str]:
    """
    Infer likely entity types based on keywords; defaults to a broad set.
    """
    lq = (query or "").lower()
    if "album" in lq:
        return ["Album"]
    if any(k in lq for k in ("artist", "band", "singer")):
        return ["Artist"]
    if any(k in lq for k in ("song", "track")):
        return ["Track"]
    if "genre" in lq:
        return ["Genre"]
    if "media" in lq:
        return ["MediaType"]
    if "playlist" in lq:
        return ["Playlist", "PlaylistTrack"]
    if "customer" in lq:
        return ["Customer"]
    if "employee" in lq:
        return ["Employee"]
    if "invoice line" in lq or "invoiceline" in lq:
        return ["InvoiceLine"]
    if "invoice" in lq:
        return ["Invoice"]
    # default broad coverage
    return [
        "Track",
        "Album",
        "Artist",
        "Genre",
        "MediaType",
        "Playlist",
        "PlaylistTrack",
        "Customer",
        "Employee",
        "Invoice",
        "InvoiceLine",
    ]


# -------------------------
# SQLite FTS5 index (stored in a separate SQLite file)
# -------------------------

def _ensure_engine(db_uri: str) -> Engine:
    return create_engine(db_uri)


def _derive_index_db_uri(source_db_uri: str) -> str:
    """
    Given a SQLite source DB URI, derive a sibling SQLite file for FTS index.
    Example: sqlite:////path/Chinook_Sqlite.sqlite -> sqlite:////path/Chinook_Sqlite_fts.sqlite
    For non-sqlite URIs, fall back to in-memory index (not persisted) — but this
    path is only used for querying, so default to a file when possible.
    """
    prefix = "sqlite:///"
    if source_db_uri.startswith(prefix):
        p = Path(source_db_uri[len(prefix) :])
        idx_path = p.with_name(f"{p.stem}_fts.sqlite")
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        return f"{prefix}{idx_path}"
    # Fallback to local file 'entity_fts.sqlite' in CWD
    fallback = Path("entity_fts.sqlite").resolve()
    return f"{prefix}{fallback}"


def ensure_fts_index(db_uri: str, index_db_uri: Optional[str] = None) -> str:
    """
    Ensure the FTS5 virtual table exists in a separate index DB and is populated.
    Returns the index DB URI used.
    """
    idx_uri = index_db_uri or _derive_index_db_uri(db_uri)
    src_engine = _ensure_engine(db_uri)
    idx_engine = _ensure_engine(idx_uri)
    with idx_engine.begin() as idx_conn:
        idx_conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts
                USING fts5(
                    entity_type,
                    entity_id,
                    name,
                    extra,
                    tokenize='porter'
                )
                """
            )
        )
        cnt = 0
        try:
            cnt = int(idx_conn.execute(text("SELECT count(*) FROM entity_fts")).scalar() or 0)
        except Exception:
            cnt = 0
        if cnt == 0:
            _populate_fts_from_source(src_engine, idx_engine)
    return idx_uri


def rebuild_fts_index(db_uri: str, index_db_uri: Optional[str] = None) -> None:
    """
    Drop and rebuild the FTS index in the separate index DB from the source tables.
    """
    idx_uri = index_db_uri or _derive_index_db_uri(db_uri)
    idx_engine = _ensure_engine(idx_uri)
    src_engine = _ensure_engine(db_uri)
    with idx_engine.begin() as idx_conn:
        idx_conn.execute(text("DROP TABLE IF EXISTS entity_fts"))
        idx_conn.execute(
            text(
                """
                CREATE VIRTUAL TABLE entity_fts
                USING fts5(
                    entity_type,
                    entity_id,
                    name,
                    extra,
                    tokenize='porter'
                )
                """
            )
        )
    _populate_fts_from_source(src_engine, idx_engine)


def _populate_fts_from_source(src_engine: Engine, idx_engine: Engine) -> None:
    """
    Populate entity_fts by reading from the source DB and inserting into the index DB.
    """
    dialect = (getattr(src_engine.url, "get_backend_name", lambda: "")() or "").lower()
    is_mysql = "mysql" in dialect

    def _cast_text(expr: str) -> str:
        return f"CAST({expr} AS CHAR)" if is_mysql else f"CAST({expr} AS TEXT)"

    def _concat_ws(sep: str, *exprs: str) -> str:
        # MySQL: CONCAT_WS(' ', a, b, c)
        if is_mysql:
            args = ", ".join(exprs)
            return f"CONCAT_WS('{sep}', {args})"
        # SQLite: a || ' ' || b || ' ' || c
        out = exprs[0]
        for e in exprs[1:]:
            out = f"{out} || '{sep}' || {e}"
        return out

    # Selects from source DB (dialect-aware for SQLite vs MySQL)
    sel_track = text(
        f"""
        SELECT
          'Track' AS entity_type,
          {_cast_text('t.TrackId')} AS entity_id,
          COALESCE(t.Name, '') AS name,
          TRIM(
            {_concat_ws(' ',
              "COALESCE(t.Composer,'')",
              "COALESCE(al.Title,'')",
              "COALESCE(ar.Name,'')",
              "COALESCE(g.Name,'')"
            )}
          ) AS extra
        FROM Track t
        LEFT JOIN Album al ON t.AlbumId = al.AlbumId
        LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
        LEFT JOIN Genre g ON t.GenreId = g.GenreId
        """
    )
    sel_album = text(
        f"""
        SELECT
          'Album' AS entity_type,
          {_cast_text('al.AlbumId')} AS entity_id,
          COALESCE(al.Title,'') AS name,
          COALESCE(ar.Name,'') AS extra
        FROM Album al
        LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
        """
    )
    sel_artist = text(
        f"""
        SELECT
          'Artist' AS entity_type,
          {_cast_text('ar.ArtistId')} AS entity_id,
          COALESCE(ar.Name,'') AS name,
          '' AS extra
        FROM Artist ar
        """
    )
    sel_genre = text(
        f"""
        SELECT
          'Genre' AS entity_type,
          {_cast_text('g.GenreId')} AS entity_id,
          COALESCE(g.Name,'') AS name,
          '' AS extra
        FROM Genre g
        """
    )
    sel_mediatype = text(
        f"""
        SELECT
          'MediaType' AS entity_type,
          {_cast_text('m.MediaTypeId')} AS entity_id,
          COALESCE(m.Name,'') AS name,
          '' AS extra
        FROM MediaType m
        """
    )
    sel_playlist = text(
        f"""
        SELECT
          'Playlist' AS entity_type,
          {_cast_text('p.PlaylistId')} AS entity_id,
          COALESCE(p.Name,'') AS name,
          '' AS extra
        FROM Playlist p
        """
    )
    # Composite key
    playlisttrack_id = (
        f"CONCAT({_cast_text('pt.PlaylistId')}, ':', {_cast_text('pt.TrackId')})"
        if is_mysql
        else f"{_cast_text('pt.PlaylistId')} || ':' || {_cast_text('pt.TrackId')}"
    )
    sel_playlisttrack = text(
        f"""
        SELECT
          'PlaylistTrack' AS entity_type,
          {playlisttrack_id} AS entity_id,
          COALESCE(t.Name,'') AS name,
          TRIM(
            {_concat_ws(' ',
              "COALESCE(p.Name,'')",
              "COALESCE(al.Title,'')",
              "COALESCE(ar.Name,'')"
            )}
          ) AS extra
        FROM PlaylistTrack pt
        LEFT JOIN Playlist p ON pt.PlaylistId = p.PlaylistId
        LEFT JOIN Track t ON pt.TrackId = t.TrackId
        LEFT JOIN Album al ON t.AlbumId = al.AlbumId
        LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
        """
    )
    sel_customer = text(
        f"""
        SELECT
          'Customer' AS entity_type,
          {_cast_text('c.CustomerId')} AS entity_id,
          TRIM({_concat_ws(' ', "COALESCE(c.FirstName,'')", "COALESCE(c.LastName,'')")}) AS name,
          TRIM({_concat_ws(' ',
            "COALESCE(c.Company,'')",
            "COALESCE(c.City,'')",
            "COALESCE(c.Country,'')",
            "COALESCE(c.Email,'')"
          )}) AS extra
        FROM Customer c
        """
    )
    sel_employee = text(
        f"""
        SELECT
          'Employee' AS entity_type,
          {_cast_text('e.EmployeeId')} AS entity_id,
          TRIM({_concat_ws(' ', "COALESCE(e.FirstName,'')", "COALESCE(e.LastName,'')")}) AS name,
          TRIM({_concat_ws(' ',
            "COALESCE(e.Title,'')",
            "COALESCE(e.City,'')",
            "COALESCE(e.Country,'')"
          )}) AS extra
        FROM Employee e
        """
    )
    invoice_name = (
        f"CONCAT('Invoice ', {_cast_text('i.InvoiceId')})"
        if is_mysql
        else f"'Invoice ' || {_cast_text('i.InvoiceId')}"
    )
    sel_invoice = text(
        f"""
        SELECT
          'Invoice' AS entity_type,
          {_cast_text('i.InvoiceId')} AS entity_id,
          {invoice_name} AS name,
          TRIM({_concat_ws(' ',
            "COALESCE(i.BillingAddress,'')",
            "COALESCE(i.BillingCountry,'')",
            "COALESCE(i.BillingCity,'')",
            "COALESCE(i.BillingState,'')",
            "COALESCE(i.BillingPostalCode,'')",
            "COALESCE(i.Total,'')"
          )}) AS extra
        FROM Invoice i
        """
    )
    invoiceline_extra = (
        f"CONCAT('Invoice ', {_cast_text('il.InvoiceId')}, ' ', COALESCE(al.Title,''), ' ', COALESCE(ar.Name,''))"
        if is_mysql
        else f"TRIM('Invoice ' || {_cast_text('il.InvoiceId')} || ' ' || COALESCE(al.Title,'') || ' ' || COALESCE(ar.Name,''))"
    )
    sel_invoiceline = text(
        f"""
        SELECT
          'InvoiceLine' AS entity_type,
          {_cast_text('il.InvoiceLineId')} AS entity_id,
          COALESCE(t.Name,'') AS name,
          TRIM({invoiceline_extra}) AS extra
        FROM InvoiceLine il
        LEFT JOIN Track t ON il.TrackId = t.TrackId
        LEFT JOIN Album al ON t.AlbumId = al.AlbumId
        LEFT JOIN Artist ar ON al.ArtistId = ar.ArtistId
        """
    )
    ins = text(
        """
        INSERT INTO entity_fts(entity_type, entity_id, name, extra)
        VALUES (:entity_type, :entity_id, :name, :extra)
        """
    )
    # Important for MySQL: a single failing SELECT can invalidate the transaction.
    # Execute each query in its own transaction and skip sources that don't match schema.
    selects = (
        sel_track,
        sel_album,
        sel_artist,
        sel_genre,
        sel_mediatype,
        sel_playlist,
        sel_playlisttrack,
        sel_customer,
        sel_employee,
        sel_invoice,
        sel_invoiceline,
    )
    for sel in selects:
        try:
            with src_engine.begin() as sconn:
                rows = [dict(r._mapping) for r in sconn.execute(sel)]
            if rows:
                with idx_engine.begin() as iconn:
                    iconn.execute(ins, rows)
        except Exception:
            # Skip tables/columns that don't exist in this DB variant.
            continue


# -------------------------
# FTS search + fuzzy rerank
# -------------------------

def _build_match(candidate: str) -> str:
    tokens = [t for t in _TOKEN_SPLIT_RE.split(candidate) if t]
    if not tokens:
        return ""
    # remove stop words and lightweight domain words
    filtered = [t for t in tokens if t.lower() not in STOP_WORDS]
    if not filtered:
        filtered = tokens
    if not tokens:
        return ""
    # Use simple AND between tokens
    # Quote each token to avoid FTS interpreting special chars
    return " AND ".join(f'"{t}"' for t in filtered)


def fts_search(
    db_uri: str,
    candidate: str,
    entity_types: Sequence[str],
    limit: int = 50,
    override_match: Optional[str] = None,
) -> List[Dict]:
    """
    Run FTS MATCH query and return raw rows with bm25 score.
    """
    if not candidate.strip() and not override_match:
        return []
    match = override_match or _build_match(candidate)
    if not match:
        return []
    type_list = list(entity_types) if entity_types else sorted(SUPPORTED_ENTITY_TYPES)
    placeholders = ", ".join(f":t{i}" for i in range(len(type_list)))
    # Always search against the index DB, not the source DB
    idx_uri = _derive_index_db_uri(db_uri)
    engine = _ensure_engine(idx_uri)
    sql = text(
        f"""
        SELECT
          entity_type,
          CAST(entity_id AS INT) AS entity_id,
          name,
          extra,
          bm25(entity_fts) AS rank
        FROM entity_fts
        WHERE entity_fts MATCH :match
          AND entity_type IN ({placeholders})
        ORDER BY rank ASC
        LIMIT :limit
        """
    )
    params_base: Dict[str, object] = {"limit": int(limit)}
    for i, tname in enumerate(type_list):
        params_base[f"t{i}"] = tname
    rows: List[Dict] = []
    with engine.begin() as conn:
        # First try: override match (phrase or OR tokens) or AND between tokens
        params = dict(params_base)
        params["match"] = match
        for r in conn.execute(sql, params):
            rows.append(dict(r._mapping))
        if rows:
            return rows
        # Fallback 1: OR between tokens
        tokens = [t for t in _TOKEN_SPLIT_RE.split(candidate) if t]
        filtered = [t for t in tokens if t.lower() not in STOP_WORDS] or tokens
        if filtered:
            or_match = " OR ".join(f'"{t}"' for t in filtered)
            params = dict(params_base)
            params["match"] = or_match
            for r in conn.execute(sql, params):
                rows.append(dict(r._mapping))
        if rows:
            return rows
        # Fallback 2: single-token probes (broad net), dedup results
        seen: set[Tuple[str, int]] = set()
        out: List[Dict] = []
        for t in filtered:
            params = dict(params_base)
            params["match"] = f'"{t}"'
            for r in conn.execute(sql, params):
                m = dict(r._mapping)
                key = (str(m.get("entity_type", "")), int(m.get("entity_id", 0)))
                if key not in seen:
                    seen.add(key)
                    out.append(m)
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
        if out:
            return out
        # Fallback 3: LIKE-based substring probes to seed candidates for fuzzy rerank
        tokens = [t for t in _TOKEN_SPLIT_RE.split(candidate) if t]
        filtered = [t for t in tokens if t.lower() not in STOP_WORDS] or tokens
        like_sql = text(
            f"""
            SELECT
              entity_type,
              CAST(entity_id AS INT) AS entity_id,
              name,
              extra,
              1000.0 AS rank
            FROM entity_fts
            WHERE entity_type IN ({placeholders})
              AND (name LIKE :pat OR extra LIKE :pat)
            LIMIT :limit
            """
        )
        seen_like: set[Tuple[str, int]] = set()
        out_like: List[Dict] = []
        # Probe with a couple of conservative patterns per token
        for tok in filtered[:3]:
            if len(tok) < 3:
                continue
            patterns = {f"%{tok[:3]}%"}
            if len(tok) >= 4:
                patterns.add(f"%{tok[:-1]}%")
            for p in patterns:
                p_params: Dict[str, object] = {"pat": p, "limit": int(limit)}
                for i, tname in enumerate(type_list):
                    p_params[f"t{i}"] = tname
                for r in conn.execute(like_sql, p_params):
                    m = dict(r._mapping)
                    key = (str(m.get("entity_type", "")), int(m.get("entity_id", 0)))
                    if key not in seen_like:
                        seen_like.add(key)
                        out_like.append(m)
                        if len(out_like) >= limit:
                            break
                if len(out_like) >= limit:
                    break
            if len(out_like) >= limit:
                break
        return out_like


def fuzzy_rerank(candidate: str, rows: List[Dict], fuzzy_candidate: Optional[str] = None) -> List[FTSCandidate]:
    """
    Compute fuzzy scores and combine with BM25 into final score.
    - Normalize bm25 across the candidate set to [0,100] with higher=better:
      bm25_norm = 100 * (max_bm25 - bm25) / (max_bm25 - min_bm25 + 1e-6)
    - final = 0.7 * fuzzy + 0.3 * bm25_norm
    """
    s = (fuzzy_candidate or candidate or "").strip()
    if normalize_for_fuzzy is not None:
        s_norm = normalize_for_fuzzy(s, remove_entity_suffixes=True)
    else:
        s_norm = _normalize_simple(s)
    try:
        lang_eff = detect_lang(s_norm) if detect_lang is not None else "en"
    except Exception:
        lang_eff = "en"
    # Note: avoid unconditional prints here; resolver has structured debug output.
    out: List[FTSCandidate] = []
    # Precompute normalization stats
    bm25_vals = [float(m.get("rank", 0.0)) for m in rows] if rows else []
    bmin = min(bm25_vals) if bm25_vals else 0.0
    bmax = max(bm25_vals) if bm25_vals else 1.0
    denom = (bmax - bmin) if (bmax - bmin) > 1e-6 else 1.0
    for m in rows:
        et = str(m.get("entity_type", ""))
        eid = int(m.get("entity_id", 0))
        name = str(m.get("name", "") or "")
        extra = str(m.get("extra", "") or "")
        bm25_score = float(m.get("rank", 0.0))
        # Robust fuzzy on normalized strings; short-circuit exact match
        if normalize_for_fuzzy is not None:
            name_norm = normalize_for_fuzzy(name, remove_entity_suffixes=True)
        else:
            name_norm = _normalize_simple(name)
        if s_norm and s_norm == name_norm:
            fuzzy_score = 100.0
        else:
            wr = float(fuzz.WRatio(s_norm, name_norm))
            if lang_eff == "en":
                # Content-token coverage penalty to avoid short unrelated matches getting 100
                q_tokens = [t for t in _TOKEN_SPLIT_RE.split(s_norm) if t and t.lower() not in STOP_WORDS]
                n_tokens = set(t for t in _TOKEN_SPLIT_RE.split(name_norm) if t)
                coverage = (sum(1 for t in q_tokens if t in n_tokens) / max(1, len(q_tokens))) if q_tokens else 0.0
                # Blend WRatio with coverage; ensure strictly < 100 when coverage < 1.0
                fuzzy_score = wr * (0.5 + 0.5 * coverage)
            else:
                # For zh/mixed, ASCII token coverage is not meaningful; rely on WRatio directly.
                fuzzy_score = wr
        # Normalize BM25 across the batch (lower bm25 is better)
        bm25_norm = 100.0 * (bmax - bm25_score) / denom
        final_score = 0.7 * fuzzy_score + 0.3 * bm25_norm
        out.append(
            FTSCandidate(
                entity_type=et,
                entity_id=eid,
                name=name,
                extra=extra,
                bm25=bm25_score,
                bm25_norm=bm25_norm,
                fuzzy=fuzzy_score,
                final=final_score,
            )
        )
    return out


# -------------------------
# Resolver orchestration
# -------------------------

def resolve_entity(
    db_uri: str,
    query: str,
    lexical_query: Optional[str] = None,
    lexical_tokens: Optional[List[str]] = None,
    intent: Optional[str] = None,
    preferred_tables: Optional[List[str]] = None,
    normalized_query: Optional[str] = None,
    fts_query_primary: Optional[str] = None,
    fts_query_fallback: Optional[str] = None,
    fuzzy_query: Optional[str] = None,
    limit: int = 50,
    debug: bool = False,
) -> Dict:
    """
    End-to-end resolution:
      - extract candidates
      - infer entity types
      - fts search + fuzzy rerank
      - decision: high / medium / low
    Returns a dict with decision, hits, primary_candidate, entity_types.
    """
    # Establish primary and fallback queries for FTS
    if fts_query_primary is not None and fts_query_primary.strip():
        primary_phrase = fts_query_primary.strip()
    elif lexical_query is not None and lexical_query.strip():
        # Use the provided lexical phrase directly as primary phrase
        primary_phrase = lexical_query.strip()
    else:
        # Default: keep stopwords for primary phrase
        tokens = [t for t in _TOKEN_SPLIT_RE.split(query or "") if t]
        primary_phrase = " ".join(tokens[:12]).strip()
    # Fallback (stopword-removed) if provided or derivable
    if fts_query_fallback is not None and fts_query_fallback.strip():
        fallback_phrase = fts_query_fallback.strip()
    else:
        toks = [t for t in _TOKEN_SPLIT_RE.split(primary_phrase) if t]
        filtered = [t for t in toks if t.lower() not in STOP_WORDS] or toks
        fallback_phrase = " ".join(filtered[:12]).strip()
    # Entity types: allow explicit mapping from preferred_tables when provided
    if preferred_tables:
        allowed = SUPPORTED_ENTITY_TYPES
        pt = [t for t in preferred_tables if t in allowed]
        # If the planner prefers only unsupported tables (e.g., Invoice), skip FTS entirely
        if not pt:
            if debug:
                try:
                    print("RESOLVER_SKIP: preferred_tables exclude FTS-supported types; skipping lexical resolver.")
                except Exception:
                    pass
            return {
                "decision": "low",
                "hits": [],
                "primary_candidate": primary_phrase,
                "entity_types": [],
            }
        types = pt
    else:
        types = infer_entity_types(query)

    # Ensure index only when we actually intend to query FTS
    ensure_fts_index(db_uri)

    # Language routing: avoid SQLite FTS MATCH for zh/mixed queries.
    try:
        lang_eff = detect_lang(query or primary_phrase) if detect_lang is not None else "en"
    except Exception:
        lang_eff = "en"

    # Build override OR match if entity_lookup and lexical tokens provided
    override_match: Optional[str] = None
    if intent == "entity_lookup" and lexical_tokens:
        toks = [t for t in lexical_tokens if t]
        if toks:
            override_match = " OR ".join(f'"{t}"' for t in toks)

    raw_rows: List[Dict] = []
    if lang_eff != "en":
        # LIKE-based lookup against the index DB to seed fuzzy rerank.
        idx_uri = _derive_index_db_uri(db_uri)
        engine = _ensure_engine(idx_uri)
        type_list = list(types) if types else sorted(SUPPORTED_ENTITY_TYPES)
        placeholders = ", ".join(f":t{i}" for i in range(len(type_list)))
        sql_like = text(
            f"""
            SELECT
              entity_type,
              CAST(entity_id AS INT) AS entity_id,
              name,
              extra,
              1000.0 AS rank
            FROM entity_fts
            WHERE entity_type IN ({placeholders})
              AND (name LIKE :pat OR extra LIKE :pat)
            LIMIT :limit
            """
        )
        pat_source = primary_phrase or fallback_phrase or (query or "")
        try:
            pat_source = (normalize_text(pat_source, lang=lang_eff) if normalize_text is not None else pat_source) or ""
        except Exception:
            pass
        pat_source = pat_source.strip()
        if pat_source:
            params: Dict[str, object] = {"pat": f"%{pat_source}%", "limit": int(limit)}
            for i, tname in enumerate(type_list):
                params[f"t{i}"] = tname
            with engine.begin() as conn:
                for r in conn.execute(sql_like, params):
                    raw_rows.append(dict(r._mapping))
        # Note: if LIKE yields nothing, we fall through to empty and decision will be low.
    else:
        # Run FTS with primary phrase preserved (phrase match), then fallback; merge/dedup
        rows_map: Dict[Tuple[str, int], Dict] = {}
        # Primary: phrase-preserving by quoting whole phrase
        if primary_phrase:
            phrase_match = f"\"{primary_phrase}\""
            primary_rows = fts_search(db_uri, primary_phrase, types, limit=limit, override_match=phrase_match)
            for r in primary_rows:
                key = (str(r.get("entity_type", "")), int(r.get("entity_id", 0)))
                if key not in rows_map:
                    rows_map[key] = r
        # Fallback: stopword-removed AND/OR logic
        if len(rows_map) < limit and fallback_phrase:
            fb_rows = fts_search(db_uri, fallback_phrase, types, limit=limit, override_match=override_match)
            for r in fb_rows:
                key = (str(r.get("entity_type", "")), int(r.get("entity_id", 0)))
                if key not in rows_map:
                    rows_map[key] = r
        raw_rows = list(rows_map.values())
    # Choose string for fuzzy comparisons
    base_norm_input = (normalized_query or query or "").strip()
    # Validate fuzzy correction: must be close to normalized query (guardrails)
    def _is_close(a: str, b: str) -> bool:
        try:
            wr = float(fuzz.WRatio(a, b))
        except Exception:
            wr = 0.0
        # Token overlap
        atoks = set(t for t in _TOKEN_SPLIT_RE.split(a) if t)
        btoks = set(t for t in _TOKEN_SPLIT_RE.split(b) if t)
        overlap = len(atoks & btoks) / max(1, len(atoks))
        return wr >= 85.0 and overlap >= 0.5
    effective_fuzzy = base_norm_input
    if fuzzy_query:
        fq = fuzzy_query.strip()
        if fq and _is_close((normalize_for_fuzzy(fq, remove_entity_suffixes=True) if normalize_for_fuzzy else _normalize_simple(fq)),
                            (normalize_for_fuzzy(base_norm_input, remove_entity_suffixes=True) if normalize_for_fuzzy else _normalize_simple(base_norm_input))):
            effective_fuzzy = fq
    fuzzy_base = effective_fuzzy or primary_phrase or query or ""
    ranked = fuzzy_rerank(fuzzy_base, raw_rows, fuzzy_candidate=effective_fuzzy)

    # Decision thresholds (with rule-based overrides)
    # Preliminary sort before thresholding (no exact priority yet)
    if ranked:
        ranked.sort(key=lambda x: (-x.final, -x.bm25_norm, -x.fuzzy, x.entity_id))
    top_fuzzy = ranked[0].fuzzy if ranked else 0.0
    decision = "low"
    if top_fuzzy >= 90.0:
        decision = "high"
    elif 75.0 <= top_fuzzy < 90.0:
        decision = "medium"

    # Rule overrides: exact match against normalized user query or fuzzy correction
    if ranked:
        # Compute normalized inputs
        if normalize_for_fuzzy is not None:
            normalized_input = normalize_for_fuzzy((normalized_query or query or ""), remove_entity_suffixes=True)
            normalized_fuzzy = normalize_for_fuzzy((fuzzy_query or ""), remove_entity_suffixes=True) if (fuzzy_query or "") else ""
        else:
            normalized_input = _normalize_simple(normalized_query or query or "")
            normalized_fuzzy = _normalize_simple(fuzzy_query or "")
        # Check any exact match across hits
        exact_idx: Optional[int] = None
        for i, h in enumerate(ranked):
            if normalize_for_fuzzy is not None:
                name_norm = normalize_for_fuzzy(h.name, remove_entity_suffixes=True)
            else:
                name_norm = _normalize_simple(h.name)
            if (normalized_input and name_norm == normalized_input) or (normalized_fuzzy and name_norm == normalized_fuzzy):
                exact_idx = i
                break
        if debug:
            try:
                print(f"FUZZY_A={normalized_input or (fuzzy_base and fuzzy_base) or ''!r}")
                # For log symmetry, show top name normalized if available
                top_name_norm = (normalize_for_fuzzy(ranked[0].name, remove_entity_suffixes=True) if normalize_for_fuzzy else _normalize_simple(ranked[0].name)) if ranked else ""
                print(f"FUZZY_B={top_name_norm!r}")
            except Exception:
                pass
        exact_pairs: set[Tuple[str, int]] = set()
        if exact_idx is not None:
            decision = "high"
            # Move exact match to front and ensure fuzzy=100, recompute final accordingly
            exact = ranked.pop(exact_idx)
            exact = FTSCandidate(
                entity_type=exact.entity_type,
                entity_id=exact.entity_id,
                name=exact.name,
                extra=exact.extra,
                bm25=exact.bm25,
                bm25_norm=exact.bm25_norm,
                fuzzy=100.0,
                final=max(exact.final, 0.7 * 100.0 + 0.3 * exact.bm25_norm),
            )
            ranked.insert(0, exact)
            exact_pairs.add((exact.entity_type, exact.entity_id))
        else:
            # Token coverage: proportion of meaningful candidate tokens present in top name
            base_norm = normalized_input or (normalize_for_fuzzy(fuzzy_base, remove_entity_suffixes=True) if normalize_for_fuzzy else _normalize_simple(fuzzy_base))
            top_name_norm2 = (normalize_for_fuzzy(ranked[0].name, remove_entity_suffixes=True) if normalize_for_fuzzy else _normalize_simple(ranked[0].name)) if ranked else ""
            cand_tokens = [t for t in _TOKEN_SPLIT_RE.split(base_norm) if t and t not in STOP_WORDS]
            name_tokens = set(t for t in _TOKEN_SPLIT_RE.split(top_name_norm2) if t)
            coverage = (sum(1 for t in cand_tokens if t in name_tokens) / max(1, len(cand_tokens))) if cand_tokens else 0.0
            if decision == "low" and coverage >= 0.6:
                decision = "medium"

    # Deterministic tie-breaking: exact-match first, then final desc, bm25_norm desc, fuzzy desc, entity_id asc
    def _sort_key(x: FTSCandidate) -> Tuple[int, float, float, float, int]:
        is_exact = 0
        try:
            if (x.entity_type, x.entity_id) in exact_pairs:
                is_exact = -1
        except Exception:
            is_exact = 0
        return (is_exact, -x.final, -x.bm25_norm, -x.fuzzy, x.entity_id)
    ranked.sort(key=_sort_key)

    # Prepare hits as plain dicts
    hits = [
        {
            "entity_type": h.entity_type,
            "entity_id": h.entity_id,
            "name": h.name,
            "extra": h.extra,
            "fuzzy": h.fuzzy,
            "bm25": h.bm25,
            "final": h.final,
        }
        for h in ranked
    ]

    if debug:
        try:
            print(f"RESOLVER_INPUTS: lexical_query={primary_phrase!r} preferred_tables={preferred_tables or []}")
            print(f"LEXICAL_NORMALIZED: {primary_phrase!r}")
            print("ENTITY_TYPES:", types)
            print("FTS_TOP5:")
            for h in hits[:5]:
                print(
                    f"  {h['entity_type']}:{h['entity_id']}  name={h['name']!r}  "
                    f"bm25={h['bm25']:.3f}  bm25_norm={next((c.bm25_norm for c in ranked if c.entity_type==h['entity_type'] and c.entity_id==h['entity_id']), 0.0):.2f}  "
                    f"fuzzy={h['fuzzy']:.1f}  final={h['final']:.2f}"
                )
            print(f"RESOLVER_DECISION: {decision}")
        except Exception:
            pass

    return {
        "decision": decision,
        "hits": hits,
        "primary_candidate": primary_phrase,
        "entity_types": types,
    }



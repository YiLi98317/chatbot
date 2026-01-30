from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .normalize import tokenize, get_stopwords
from .bm25 import search_bm25, BM25Hit


# Lightweight synonym map focused on the Chinook/music domain
SYNONYMS: Dict[str, List[str]] = {
    "song": ["track"],
    "tracks": ["songs"],
    "track": ["song"],
    "album": ["record"],
    "artist": ["band", "singer"],
    "band": ["artist"],
    "singer": ["artist", "vocalist"],
    "title": ["name"],
    "named": ["called", "titled"],
    "called": ["titled", "named"],
}


@dataclass(frozen=True)
class ExpansionResult:
    expanded_query: str
    added_terms: List[str]


def _expand_with_synonyms(tokens: List[str], max_added: int = 5) -> List[str]:
    seen = set(t.lower() for t in tokens)
    added: List[str] = []
    for t in tokens:
        for s in SYNONYMS.get(t.lower(), []):
            if s not in seen:
                added.append(s)
                seen.add(s)
            if len(added) >= max_added:
                return added
    return added


def deterministic_expand(query: str, *, max_added: int = 5) -> ExpansionResult:
    """
    Deterministic expansion using a small synonym map and simple n-gram stitching.
    """
    try:
        sw = get_stopwords()
    except Exception:
        sw = set()
    toks = [t for t in tokenize(query) if t and t.lower() not in sw]
    if not toks:
        return ExpansionResult(expanded_query=query, added_terms=[])
    # Synonym additions first
    added = _expand_with_synonyms(toks, max_added=max_added)
    expanded = " ".join([t for t in (query.strip(), " ".join(added).strip()) if t])
    return ExpansionResult(expanded_query=expanded, added_terms=added)


def apply_qexp_bm25(
    db_uri: str,
    query: str,
    *,
    entity_types: Optional[Sequence[str]] = None,
    top_k: int = 50,
) -> List[Dict]:
    """
    Apply deterministic expansion then run BM25 with the expanded query.
    """
    exp = deterministic_expand(query, max_added=5)
    hits = search_bm25(db_uri=db_uri, query=exp.expanded_query, entity_types=entity_types, limit=top_k)
    out: List[Dict] = []
    for h in hits:
        out.append(
            {
                "text": h.name,
                "score": float(h.score),
                "metadata": {
                    "source": "bm25_qexp",
                    "added_terms": exp.added_terms,
                    "bm25_raw": float(h.bm25),
                    "entity_type": h.entity_type,
                    "entity_id": h.entity_id,
                    "name": h.name,
                    "extra": h.extra,
                },
            }
        )
    return out


# Placeholder for optional LLM-assisted expansion with strict budget guard.
# Intentionally returns the deterministic expansion unless explicitly extended.
def llm_assisted_expand(
    query: str,
    *,
    enable_llm: bool = False,
    max_tokens_added: int = 8,
) -> ExpansionResult:
    """
    Stub for LLM-assisted expansion; disabled by default for determinism.
    """
    return deterministic_expand(query, max_added=min(5, max_tokens_added))

from typing import Dict, List, Sequence, Tuple
from chatbot.retrieval.lexical import tokenize as lexical_tokenize
from chatbot.retrieval.bm25 import bm25_search


_SYNONYMS: Dict[str, List[str]] = {
    "song": ["track"],
    "songs": ["tracks"],
    "track": ["song"],
    "tracks": ["songs"],
    "artist": ["singer", "band"],
    "artists": ["singers", "bands"],
    "album": ["record", "lp"],
    "albums": ["records", "lps"],
    "title": ["name"],
    "genre": ["style"],
}


def _generate_bigrams(tokens: Sequence[str]) -> List[str]:
    if not tokens:
        return []
    out: List[str] = []
    for i in range(len(tokens) - 1):
        a = tokens[i].strip()
        b = tokens[i + 1].strip()
        if a and b:
            out.append(f"{a} {b}")
    return out


def deterministic_query_expansion(
    query: str,
    *,
    max_new_terms: int = 4,
    include_bigrams: bool = True,
) -> Tuple[str, List[str]]:
    """
    Deterministic expansion using simple synonym map and optional bigrams.
    Returns (expanded_query, added_terms).
    """
    base_tokens = [t.lower() for t in lexical_tokenize(query)]
    if not base_tokens:
        return (query, [])
    added: List[str] = []
    seen = set(base_tokens)
    # Synonyms
    for t in base_tokens:
        for syn in _SYNONYMS.get(t, []):
            if syn not in seen:
                added.append(syn)
                seen.add(syn)
            if len(added) >= max_new_terms:
                break
        if len(added) >= max_new_terms:
            break
    # Optional bigrams
    if include_bigrams and len(added) < max_new_terms:
        for bg in _generate_bigrams(base_tokens):
            if bg not in seen:
                added.append(bg)
                seen.add(bg)
            if len(added) >= max_new_terms:
                break
    if not added:
        return (query, [])
    expanded = (query.strip() + " " + " ".join(added)).strip()
    return (expanded, added)


def expanded_search(
    db_uri: str,
    query: str,
    preferred_tables: List[str],
    *,
    limit: int = 10,
    debug: bool = False,
    allow_llm: bool = False,
) -> List[Dict]:
    """
    Run deterministic expansion then BM25 search.
    LLM-assisted expansion is intentionally disabled by default and subject to budget guardrails.
    """
    expanded, added = deterministic_query_expansion(query)
    if debug:
        try:
            print("QEXP_ADDED_TERMS:", added)
            print("QEXP_EXPANDED_QUERY:", repr(expanded))
        except Exception:
            pass
    results = bm25_search(db_uri=db_uri, query=expanded, preferred_tables=preferred_tables, limit=limit, debug=debug)
    for d in results:
        meta = d.get("metadata") or {}
        meta["qexp_added_terms"] = added
        d["metadata"] = meta
    return results


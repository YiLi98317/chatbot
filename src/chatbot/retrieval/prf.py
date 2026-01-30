from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .bm25 import search_bm25, BM25Hit
from .bm25 import bm25_search
from .normalize import tokenize, get_stopwords
from chatbot.retrieval.lexical import tokenize as lexical_tokenize


@dataclass(frozen=True)
class PRFResult:
    expanded_query: str
    expansion_tokens: List[str]
    seed_hits: List[BM25Hit]


def _select_expansion_tokens(
    hits: List[BM25Hit],
    *,
    max_tokens: int = 5,
    exclude: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Deterministically select expansion tokens from top BM25 hits.
    Strategy:
      - tokenize name and extra fields
      - drop stopwords and very short tokens (<3)
      - weight by document rank (higher weight for earlier hits)
      - sum weights across docs and return top-N not in exclude
    """
    exclude_set = set(t.lower() for t in (exclude or []))
    try:
        sw = get_stopwords()
    except Exception:
        sw = set()
    weights: Dict[str, float] = defaultdict(float)
    for rank_idx, h in enumerate(hits):
        # Higher weight for early hits (1.0, 0.9, 0.8, ...)
        w = 1.0 / (1.0 + rank_idx * 0.1)
        for field in (h.name, h.extra):
            for tok in tokenize(field):
                tl = tok.lower()
                if len(tl) < 3:
                    continue
                if tl in sw:
                    continue
                if tl in exclude_set:
                    continue
                weights[tl] += w
    # Sort by weight desc then lexicographically for determinism
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[: max(0, int(max_tokens))]]


def rocchio_prf(
    db_uri: str,
    query: str,
    *,
    entity_types: Optional[Sequence[str]] = None,
    seed_k: int = 10,
    expansion_tokens: int = 5,
) -> PRFResult:
    """
    Deterministic, lightweight PRF:
      1) Run BM25 to fetch top seed_k hits
      2) Extract weighted tokens from hits
      3) Return expanded query = query + selected tokens
    """
    seed_hits = search_bm25(db_uri=db_uri, query=query, entity_types=entity_types, limit=seed_k)
    # Tokens present in the original query
    orig_tokens = [t.lower() for t in tokenize(query)]
    exp_toks = _select_expansion_tokens(seed_hits, max_tokens=expansion_tokens, exclude=orig_tokens)
    expanded = " ".join([t for t in (query.strip(), " ".join(exp_toks).strip()) if t])
    return PRFResult(expanded_query=expanded, expansion_tokens=exp_toks, seed_hits=seed_hits)


def apply_prf_bm25(
    db_uri: str,
    query: str,
    *,
    entity_types: Optional[Sequence[str]] = None,
    seed_k: int = 10,
    expansion_tokens: int = 5,
    top_k: int = 50,
) -> List[Dict]:
    """
    One-shot helper:
      - compute PRF-expanded query
      - rerun BM25 with the expanded query
      - return normalized dicts for downstream merging
    """
    prf = rocchio_prf(
        db_uri=db_uri,
        query=query,
        entity_types=entity_types,
        seed_k=seed_k,
        expansion_tokens=expansion_tokens,
    )
    # Use expanded query for a broader BM25 pass
    hits = search_bm25(db_uri=db_uri, query=prf.expanded_query, entity_types=entity_types, limit=top_k)
    out: List[Dict] = []
    for h in hits:
        out.append(
            {
                "text": h.name,
                "score": float(h.score),
                "metadata": {
                    "source": "bm25_prf",
                    "expansion_tokens": prf.expansion_tokens,
                    "bm25_raw": float(h.bm25),
                    "entity_type": h.entity_type,
                    "entity_id": h.entity_id,
                    "name": h.name,
                    "extra": h.extra,
                },
            }
        )
    return out

from collections import Counter


def _top_terms_from_titles(
    top_docs: Sequence[Dict],
    *,
    exclude: Sequence[str],
    max_terms: int,
) -> List[str]:
    """
    Extract high-signal expansion terms from document titles/texts.
    Heuristic: frequency across top documents; exclude original query terms.
    """
    ex = set(t.lower() for t in exclude)
    counts: Counter[str] = Counter()
    for d in top_docs:
        text = (d.get("text") or "").strip()
        if not text:
            continue
        toks = [t.lower() for t in lexical_tokenize(text)]
        for t in toks:
            if t and t not in ex:
                counts[t] += 1
    # Prefer multi-occurrence terms, break ties by term order deterministically
    items = list(counts.items())
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in items[: max(0, int(max_terms))]]


def prf_expand_query(
    original_query: str,
    top_docs: Sequence[Dict],
    *,
    max_terms: int = 3,
) -> str:
    """
    Deterministic PRF expansion: pick top frequent tokens from top docs
    not present in original query tokens and append to the query.
    """
    orig_terms = [t.lower() for t in lexical_tokenize(original_query)]
    expand_terms = _top_terms_from_titles(top_docs, exclude=orig_terms, max_terms=max_terms)
    if not expand_terms:
        return original_query
    return (original_query.strip() + " " + " ".join(expand_terms)).strip()


def prf_search(
    db_uri: str,
    query: str,
    preferred_tables: List[str],
    *,
    initial_k: int = 10,
    max_terms: int = 3,
    limit: int = 10,
    debug: bool = False,
) -> List[Dict]:
    """
    Pseudo-Relevance Feedback:
      1) Run BM25 to collect top-k seeds
      2) Extract frequent tokens from seeds (excluding original tokens)
      3) Re-run BM25 with expanded query for final results
    """
    if not query or not query.strip():
        return []
    seeds = bm25_search(db_uri=db_uri, query=query, preferred_tables=preferred_tables, limit=max(1, initial_k), debug=debug)
    if debug:
        try:
            print("PRF_SEEDS_COUNT:", len(seeds))
        except Exception:
            pass
    expanded = prf_expand_query(query, seeds, max_terms=max_terms)
    if debug:
        try:
            print("PRF_EXPANDED_QUERY:", repr(expanded))
        except Exception:
            pass
    final = bm25_search(db_uri=db_uri, query=expanded, preferred_tables=preferred_tables, limit=limit, debug=debug)
    # Mark provenance
    for d in final:
        meta = d.get("metadata") or {}
        meta["prf_expanded"] = True
        meta["prf_source_query"] = query
        d["metadata"] = meta
    return final


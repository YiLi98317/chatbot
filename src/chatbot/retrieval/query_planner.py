from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


def _plan_log(msg: str, debug_only: bool = False, debug: bool = False) -> None:
    if debug_only and not debug:
        return
    try:
        print(f"[plan] {msg}", flush=True)
    except Exception:
        pass


@dataclass(frozen=True)
class QueryPlan:
    raw_query: str
    intent: str  # "entity_lookup" | "semantic" | "mixed"
    entity_candidates: List[str]
    preferred_tables: List[str]
    lexical_query: str
    vector_query: str
    # Normalization/post-processing fields
    # Always preserve the exact user input captured at the boundary (CLI/HTTP)
    user_input_raw: str = ""
    # LLM-proposed alternatives retained for reference but never used as primary
    query_variants: List[str] = field(default_factory=list)
    lexical_tokens: List[str] = field(default_factory=list)
    fuzzy_candidate: str = ""
    vector_query_clean: str = ""
    # Additional explicit variants for transparency/debugging
    query_normalized: str = ""          # raw_query normalized (casefold + punctuation normalized)
    query_for_fts_primary: str = ""     # phrase-preserving normalized text (no stopword drop)
    query_for_fts_fallback: str = ""    # stopword-removed variant for fallback matching
    query_for_fuzzy: str = ""           # string used for fuzzy comparisons
    filters: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None


SUPPORTED_TABLES_FALLBACK = [
    "Track",
    "Album",
    "Artist",
    "Customer",
    "Employee",
    "Invoice",
    "InvoiceLine",
    "Playlist",
    "Genre",
    "MediaType",
]

_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zA-Z']+")
_WRAPPER_WORDS = {
    "tell",
    "give",
    "info",
    "about",
    "please",
    "can",
    "could",
    "would",
    "do",
    "you",
    "know",
    "is",
    "there",
    "a",
    "an",
    "the",
    "me",
    "my",
    "your",
    "list",
    "show",
}
_STOP_WORDS = _WRAPPER_WORDS.union(
    {
        "in",
        "on",
        "for",
        "from",
        "and",
        "or",
        "of",
        "named",
        "called",
        "titled",
        "song",
        "track",
        "album",
        "artist",
        "style",
        "tracks",
        "songs",
        "albums",
        "artists",
    }
)


def _ollama_generate_json(prompt: str, model: str, base_url: str, temperature: float = 0.1, num_predict: int = 256) -> str:
    """
    Call Ollama /api/generate with options tuned for planning (low temperature, short output).
    Returns raw response text (expected to be JSON string).
    """
    url = f"{base_url.rstrip('/')}/api/generate"
    resp = requests.post(
        url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Ask Ollama to enforce JSON output when supported.
            "format": "json",
            "options": {
                "temperature": max(0.0, min(1.0, float(temperature))),
                "num_predict": int(num_predict),
            },
        },
        timeout=300,  # CPU inference (e.g. DeepSeek R1) can exceed 60s
    )
    resp.raise_for_status()
    data = resp.json()
    out = data.get("response")
    if not isinstance(out, str):
        raise RuntimeError(f"Unexpected planner response: {data}")
    if not out.strip():
        raise RuntimeError(f"Planner returned empty response: {data}")
    return out.strip()

def _extract_json_object_text(raw_out: str) -> str:
    """
    Best-effort extraction of a JSON object string from model output.
    Handles common cases:
    - empty/whitespace
    - ```json ... ``` fences
    - leading/trailing commentary
    """
    s = (raw_out or "").strip()
    if not s:
        return ""
    # Strip markdown fences if present
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
        s = re.sub(r"\s*```$", "", s).strip()
    # If model adds text, take the first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1].strip()
    return s


def _build_planner_prompt(raw_query: str, supported_tables: Sequence[str]) -> str:
    """
    Build an instruction + few-shot prompt that MUST emit JSON only.
    """
    supported = ", ".join(supported_tables)
    return (
        "You are a Query Planner. Output STRICT JSON ONLY. No commentary.\n"
        "Schema:\n"
        '{\n'
        '  "raw_query": string,\n'
        '  "intent": "entity_lookup" | "semantic" | "mixed",\n'
        '  "entity_candidates": string[],\n'
        f'  "preferred_tables": string[] subset of [{supported}],\n'
        '  "lexical_query": string,\n'
        '  "vector_query": string,\n'
        '  "filters": object,\n'
        '  "needs_clarification": boolean,\n'
        '  "clarification_question": string\n'
        '}\n'
        "Rules:\n"
        "- Keep lexical_query short and entity-focused.\n"
        "- Keep vector_query concise; remove wrappers like 'do you know' or 'tell me about'.\n"
        "- If user asks about a specific named thing, set intent=entity_lookup and include that phrase in entity_candidates.\n"
        "- If the query is vague/semantic, set intent=semantic and focus vector_query accordingly.\n"
        "- Never invent database facts; only rewrite the query and propose filters.\n\n"
        "Examples:\n"
        "User: Do you know the song fly me to the mooon?\n"
        '{\n'
        '  "raw_query": "Do you know the song fly me to the mooon?",\n'
        '  "intent": "entity_lookup",\n'
        '  "entity_candidates": ["fly me to the mooon"],\n'
        '  "preferred_tables": ["Track"],\n'
        '  "lexical_query": "fly me to the mooon",\n'
        '  "vector_query": "Fly Me To The Moon track",\n'
        '  "filters": {"table": ["Track"]},\n'
        '  "needs_clarification": false,\n'
        '  "clarification_question": ""\n'
        '}\n'
        "User: Frank Sinatra style tracks\n"
        '{\n'
        '  "raw_query": "Frank Sinatra style tracks",\n'
        '  "intent": "semantic",\n'
        '  "entity_candidates": [],\n'
        '  "preferred_tables": ["Track"],\n'
        '  "lexical_query": "Frank Sinatra",\n'
        '  "vector_query": "tracks similar to Frank Sinatra style: vocal jazz, traditional pop, swing",\n'
        '  "filters": {"table": ["Track"]},\n'
        '  "needs_clarification": false,\n'
        '  "clarification_question": ""\n'
        '}\n'
        "User: List tracks from album Back In Black\n"
        '{\n'
        '  "raw_query": "List tracks from album Back In Black",\n'
        '  "intent": "mixed",\n'
        '  "entity_candidates": ["Back In Black"],\n'
        '  "preferred_tables": ["Album", "Track"],\n'
        '  "lexical_query": "Back In Black",\n'
        '  "vector_query": "album Back In Black track list",\n'
        '  "filters": {"table": ["Album", "Track"]},\n'
        '  "needs_clarification": false,\n'
        '  "clarification_question": ""\n'
        '}\n'
        f"User: {raw_query}\n"
    )


def _validate_and_coerce(obj: Dict[str, Any], supported_tables: Sequence[str], raw: str) -> Optional[QueryPlan]:
    try:
        # Ignore any planner-provided raw_query; preserve true user input
        raw_query_val = str(raw or "").strip()
        intent = obj.get("intent")
        if intent not in ("entity_lookup", "semantic", "mixed"):
            return None
        entity_candidates = obj.get("entity_candidates") or []
        if not isinstance(entity_candidates, list):
            return None
        entity_candidates = [str(x).strip() for x in entity_candidates if str(x).strip()]
        if len(entity_candidates) > 5:
            entity_candidates = entity_candidates[:5]
        preferred_tables = obj.get("preferred_tables") or []
        if not isinstance(preferred_tables, list):
            return None
        preferred_tables = [t for t in preferred_tables if t in set(supported_tables)]
        lexical_query = str(obj.get("lexical_query") or "").strip()
        vector_query = str(obj.get("vector_query") or "").strip()
        filters = obj.get("filters") or {}
        if not isinstance(filters, dict):
            filters = {}
        needs_clarification = bool(obj.get("needs_clarification", False))
        clarification_question_raw = obj.get("clarification_question")
        clarification_question = (
            str(clarification_question_raw).strip() if clarification_question_raw is not None else None
        )
        # Minimal content requirements
        if not lexical_query and entity_candidates:
            lexical_query = entity_candidates[0]
        if not vector_query:
            vector_query = lexical_query or (entity_candidates[0] if entity_candidates else "")
        return QueryPlan(
            raw_query=raw_query_val,
            user_input_raw=raw_query_val,
            intent=intent,
            entity_candidates=entity_candidates,
            preferred_tables=preferred_tables,
            lexical_query=lexical_query,
            vector_query=vector_query,
            filters=filters,
            needs_clarification=needs_clarification,
            clarification_question=clarification_question,
        )
    except Exception:
        return None


def _infer_tables_from_text(text: str, supported_tables: Sequence[str]) -> List[str]:
    lq = (text or "").lower()
    order: List[str] = []
    if "album" in lq:
        order.extend([t for t in ("Album", "Track", "Artist") if t in supported_tables])
    elif "artist" in lq or "band" in lq:
        order.extend([t for t in ("Artist", "Track", "Album") if t in supported_tables])
    elif "track" in lq or "song" in lq or "playlist" in lq:
        order.extend([t for t in ("Track", "Album", "Artist", "Playlist") if t in supported_tables])
    else:
        order.extend([t for t in ("Track", "Album", "Artist") if t in supported_tables])
    # Ensure unique and keep only supported
    seen = set()
    uniq = []
    for t in order:
        if t in supported_tables and t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def _extract_quoted_phrase(s: str) -> Optional[str]:
    if not s:
        return None
    m = re.search(r'"([^"]{2,})"', s)
    if m:
        return m.group(1).strip()
    m = re.search(r"'([^']{2,})'", s)
    if m:
        return m.group(1).strip()
    return None


def _tokenize(s: str) -> List[str]:
    return [t for t in _TOKEN_SPLIT_RE.split(s or "") if t]


def _looks_like_title(s: str) -> bool:
    # Heuristic for "this is probably a named thing":
    # - Title-case (multiple capitalized words), OR
    # - Short, noun-phrase-like input (even if lowercase), e.g. song/album titles.
    if not s or len(s) > 128:
        return False
    words = s.split()
    cap_count = sum(1 for w in words if w[:1].isupper())
    if cap_count >= max(2, len(words) // 2):
        return True

    # Lowercase titles: avoid classifying real questions as titles.
    if any(ch in s for ch in ("?", "？")):
        return False
    # Too short/too long are unlikely to be stable titles.
    if not (3 <= len(words) <= 8):
        return False
    first = (words[0] or "").lower()
    if first in {
        "what",
        "who",
        "how",
        "why",
        "where",
        "when",
        "tell",
        "give",
        "list",
        "show",
        "do",
        "does",
        "did",
        "is",
        "are",
        "can",
        "could",
        "would",
    }:
        return False
    return True


def deterministic_fallback_plan(raw_query: str, supported_tables: Sequence[str] | None = None) -> QueryPlan:
    """
    Deterministic fallback planning without regex wrapper lists.
    """
    if not supported_tables:
        supported_tables = SUPPORTED_TABLES_FALLBACK
    supported_tables = list(supported_tables)
    quoted = _extract_quoted_phrase(raw_query)
    tokens = _tokenize(raw_query)
    content_tokens = [t for t in tokens if t.lower() not in _STOP_WORDS]
    lexical_query = " ".join(content_tokens[:12]).strip()
    if quoted:
        lexical_query = quoted
    # Decide vector query
    wrapper_ratio = 0.0
    if tokens:
        wrapper_ratio = sum(1 for t in tokens if t.lower() in _WRAPPER_WORDS) / max(1, len(tokens))
    vector_query = raw_query if wrapper_ratio < 0.35 else (lexical_query or raw_query)
    # Intent
    intent = "entity_lookup" if quoted or _looks_like_title(raw_query.strip()) else "semantic"
    preferred_tables = _infer_tables_from_text(raw_query, supported_tables)
    filters: Dict[str, Any] = {}
    if preferred_tables:
        filters["table"] = [t for t in preferred_tables if t in supported_tables]
    return QueryPlan(
        raw_query=raw_query,
        user_input_raw=raw_query,
        intent=intent,
        entity_candidates=[lexical_query] if lexical_query else [],
        preferred_tables=preferred_tables,
        lexical_query=lexical_query,
        vector_query=vector_query or lexical_query,
        filters=filters,
        needs_clarification=False,
        clarification_question=None,
    )


_ENTITY_SUFFIXES = {
    "track",
    "tracks",
    "song",
    "songs",
    "album",
    "albums",
    "artist",
    "artists",
    "playlist",
    "genre",
    "mediatype",
    "customer",
    "employee",
    "invoice",
    "invoiceline",
}


# Use NLTK English stopwords when available; fall back to empty set
try:
    import importlib
    _nltk_corpus = importlib.import_module("nltk.corpus")
    try:
        _EN_STOPWORDS = set(w.lower() for w in _nltk_corpus.stopwords.words("english"))
    except Exception:
        _EN_STOPWORDS = set()
except Exception:
    _EN_STOPWORDS = set()

_TRAILING_JUNK_WORDS = _ENTITY_SUFFIXES.union(_EN_STOPWORDS)


def _strip_entity_suffixes(s: str) -> str:
    parts = (s or "").strip().split()
    while parts and parts[-1].lower() in _TRAILING_JUNK_WORDS:
        parts = parts[:-1]
    return " ".join(parts).strip()


def _post_process_plan(plan: QueryPlan) -> QueryPlan:
    """
    Overwrite planner-provided fields with deterministic normalization outputs.
    """
    try:
        from .normalize import normalize_for_lexical, normalize_for_fuzzy
    except Exception:
        # If normalize module is unavailable for any reason, return plan as-is
        return plan

    raw_query = plan.raw_query or ""
    # For entity_lookup, primary text comes from the literal user input (not LLM guesses)
    if plan.intent == "entity_lookup":
        base_text = raw_query
    else:
        # For non-entity, allow planner lexical/vector hints
        base_text = plan.lexical_query or raw_query

    # Lexical normalization (tokens and query)
    # Primary variant for FTS should keep stopwords; fallback removes them
    lq_primary, _ = normalize_for_lexical(base_text, remove_stopwords=False, remove_entity_suffixes=True)
    lq_fallback, ltokens = normalize_for_lexical(base_text, remove_stopwords=True, remove_entity_suffixes=True)
    if not (lq_primary or lq_fallback) and (plan.lexical_query or "").strip():
        # fallback to normalized version of planner lexical if base_text got emptied
        lq_primary, _ = normalize_for_lexical(plan.lexical_query, remove_stopwords=False, remove_entity_suffixes=True)
        lq_fallback, ltokens = normalize_for_lexical(plan.lexical_query, remove_stopwords=True, remove_entity_suffixes=True)

    # Fuzzy normalization (string)
    fuzzy_cand = normalize_for_fuzzy(base_text, remove_entity_suffixes=True)

    # Vector query selection
    vq = plan.vector_query or raw_query
    vq_clean = normalize_for_fuzzy(vq, remove_entity_suffixes=False)
    # Normalized user query (no entity suffix removal)
    q_norm = normalize_for_fuzzy(raw_query, remove_entity_suffixes=False)

    # Collect LLM-proposed variants without mutating primaries
    variants: List[str] = []
    for x in (plan.lexical_query, plan.vector_query) + tuple(plan.entity_candidates or []):
        if isinstance(x, str) and x.strip():
            variants.append(x.strip())
    # Dedupe while preserving order
    seen: set[str] = set()
    query_variants = []
    for v in variants:
        key = v.lower()
        if key not in seen:
            seen.add(key)
            query_variants.append(v)

    return QueryPlan(
        raw_query=plan.raw_query,
        user_input_raw=plan.user_input_raw or raw_query,
        intent=plan.intent,
        entity_candidates=plan.entity_candidates,
        preferred_tables=plan.preferred_tables,
        # Ensure lexical_query reflects normalized literal input for entity lookups
        lexical_query=lq_fallback if plan.intent == "entity_lookup" else (lq_fallback or (plan.lexical_query or "")),
        vector_query=vq,
        query_variants=query_variants,
        lexical_tokens=ltokens,
        fuzzy_candidate=fuzzy_cand,
        vector_query_clean=vq_clean,
        query_normalized=q_norm,
        query_for_fts_primary=lq_primary,
        query_for_fts_fallback=lq_fallback,
        query_for_fuzzy=fuzzy_cand,
        filters=plan.filters,
        needs_clarification=plan.needs_clarification,
        clarification_question=plan.clarification_question,
    )


def plan_query(
    raw_query: str,
    ollama_base_url: str,
    model: str,
    supported_tables: Sequence[str] | None = None,
    enable_llm: bool = True,
    debug: bool = False,
) -> QueryPlan:
    """
    Plan a query using the LLM, validate JSON, and fall back deterministically on any failure.
    """
    if not supported_tables:
        supported_tables = SUPPORTED_TABLES_FALLBACK
    supported_tables = list(supported_tables)
    if not enable_llm:
        _plan_log("LLM disabled, using deterministic fallback", debug=False, debug_only=False)
        return deterministic_fallback_plan(raw_query, supported_tables)
    try:
        _plan_log("Building planner prompt...", debug_only=False, debug=False)
        t0 = time.time()
        prompt = _build_planner_prompt(raw_query, supported_tables)
        _plan_log(f"Prompt built in {time.time() - t0:.2f}s ({len(prompt)} chars), calling Ollama (model={model}, base_url={ollama_base_url})...", debug_only=False, debug=False)
        sys.stdout.flush()
        t1 = time.time()
        raw_out = _ollama_generate_json(prompt=prompt, model=model, base_url=ollama_base_url, temperature=0.1, num_predict=256)
        elapsed = time.time() - t1
        _plan_log(f"Ollama returned in {elapsed:.2f}s, parsing JSON...", debug_only=False, debug=False)
        if debug:
            try:
                print("PLANNER_RAW_OUTPUT:")
                print(raw_out)
                print("PLANNER_RAW_LEN:", len(raw_out or ""))
            except Exception:
                pass
        json_text = _extract_json_object_text(raw_out)
        if debug and (json_text or "") != (raw_out or ""):
            try:
                print("PLANNER_JSON_EXTRACTED:")
                print(json_text)
            except Exception:
                pass
        obj = json.loads(json_text)
        if not isinstance(obj, dict):
            return deterministic_fallback_plan(raw_query, supported_tables)
        plan = _validate_and_coerce(obj, supported_tables, raw_query)
        if plan is None:
            return deterministic_fallback_plan(raw_query, supported_tables)
        # Ensure preferred_tables not empty if filters specify table
        if not plan.preferred_tables and isinstance(plan.filters.get("table"), list):
            pt = [t for t in plan.filters.get("table", []) if t in supported_tables]
            if pt:
                plan = QueryPlan(
                    raw_query=plan.raw_query,
                    intent=plan.intent,
                    entity_candidates=plan.entity_candidates,
                    preferred_tables=pt,
                    lexical_query=plan.lexical_query,
                    vector_query=plan.vector_query,
                    lexical_tokens=plan.lexical_tokens,
                    fuzzy_candidate=plan.fuzzy_candidate,
                    vector_query_clean=plan.vector_query_clean,
                    filters=plan.filters,
                    needs_clarification=plan.needs_clarification,
                    clarification_question=plan.clarification_question,
                )
        # Deprecate legacy trailing trim by default; leave behind flag for emergencies
        use_legacy_trim = False
        try:
            from chatbot.config import get_settings  # type: ignore

            use_legacy_trim = bool(getattr(get_settings(), "enable_legacy_trailing_trim", False))
        except Exception:
            use_legacy_trim = False
        if use_legacy_trim and plan.intent == "entity_lookup" and plan.vector_query:
            corrected = _strip_entity_suffixes(plan.vector_query)
            if corrected:
                plan = QueryPlan(
                    raw_query=plan.raw_query,
                    intent=plan.intent,
                    entity_candidates=plan.entity_candidates,
                    preferred_tables=plan.preferred_tables,
                    lexical_query=corrected,
                    vector_query=plan.vector_query,
                    lexical_tokens=plan.lexical_tokens,
                    fuzzy_candidate=plan.fuzzy_candidate,
                    vector_query_clean=plan.vector_query_clean,
                    filters=plan.filters,
                    needs_clarification=plan.needs_clarification,
                    clarification_question=plan.clarification_question,
                )
        # Always perform deterministic normalization post-processing
        _plan_log("Plan validated, running post-process...", debug_only=True, debug=debug)
        return _post_process_plan(plan)
    except Exception as e:
        _plan_log(f"Planner error: {type(e).__name__}: {e}, using fallback", debug_only=False, debug=False)
        if debug:
            try:
                print("PLANNER_ERROR:", type(e).__name__, str(e), flush=True)
                print("PLANNER_FALLBACK_USED: true", flush=True)
            except Exception:
                pass
        return _post_process_plan(deterministic_fallback_plan(raw_query, supported_tables))


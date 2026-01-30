from __future__ import annotations

import json
from dataclasses import asdict
from typing import List, Dict, Optional

from qdrant_client import QdrantClient

from chatbot.config import get_settings
from chatbot.retrieval.query_planner import plan_query, deterministic_fallback_plan
from chatbot.retrieval.retriever import retrieve_top_k
from chatbot.vectorstore.qdrant_store import QdrantStore
from chatbot.retrieval.normalize import get_stopwords


def run_case(label: str, question: str, debug: bool = True) -> None:
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    store = QdrantStore(client, base_url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    print(f"\n=== {label}. {question}")
    plan = plan_query(
        raw_query=question,
        ollama_base_url=settings.ollama_base_url,
        model=settings.chat_model,
        supported_tables=[
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
        ],
        enable_llm=settings.enable_query_planner,
    )
    print("QUERY_PLAN:", json.dumps(asdict(plan), indent=2))
    # Derived base_text consistent with normalization
    if plan.intent == "entity_lookup" and plan.entity_candidates:
        base_text = plan.entity_candidates[0]
    else:
        base_text = plan.lexical_query or question
    print("BASE_TEXT:", repr(base_text))
    print("LEXICAL_QUERY:", repr(plan.lexical_query))
    print("LEXICAL_TOKENS:", getattr(plan, "lexical_tokens", []))
    print("FUZZY_CANDIDATE:", repr(getattr(plan, "fuzzy_candidate", "")))

    results: List[Dict] = retrieve_top_k(
        store=store,
        collection=settings.default_collection,
        query=question,
        embed_model=settings.embed_model,
        ollama_base_url=settings.ollama_base_url,
        top_k=settings.default_top_k,
        db_uri=settings.db_uri,
        debug=debug,
    )
    if not results:
        print("No results.")
        return
    meta0 = results[0].get("metadata", {}) or {}
    if meta0.get("resolver_decision") == "medium":
        print("Resolver MEDIUM: Suggestions:")
        for s in meta0.get("suggestions", [])[:3]:
            print(" -", s.get("entity_type"), s.get("name"), f"(id={s.get('entity_id')})")
    elif meta0.get("resolver_decision") == "low_with_qdrant":
        print("Resolver LOW; Qdrant closest matches:")
        for m in meta0.get("matches", [])[:3]:
            print(" -", m.get("table"), m.get("title"))
    else:
        print(f"Top {len(results)} contexts:")
        for i, r in enumerate(results, 1):
            src = r.get("metadata", {}) or {}
            table = src.get("table", "")
            title = src.get("title") or src.get("Name") or src.get("TrackId") or ""
            print(f" {i}.", table, title, f"(score={r.get('score'):.4f})")


def main() -> None:
    cases = [
        ("1", "do you know about avril alvin"),
        ("2", "do you know the song fly me to the moon"),
        ("3", "is there a track named fly somebody to the mooon"),
        ("4", "the the"),
    ]
    for label, q in cases:
        run_case(label, q)
    # Case 5: NLTK stopwords presence check
    print("\n=== 5. NLTK stopwords resource check")
    try:
        sw = get_stopwords()
        print(f"STOPWORDS_OK: size={len(sw)}")
    except Exception as e:
        print(f"STOPWORDS_ERROR: {e}")


if __name__ == "__main__":
    main()


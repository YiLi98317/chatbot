from __future__ import annotations

from typing import List

from chatbot.config import Settings
from chatbot.llm.ollama_chat import generate
from chatbot.retrieval.retriever import retrieve_top_k
from chatbot.vectorstore.qdrant_store import QdrantStore
from chatbot.retrieval.normalize import detect_lang


def build_prompt(question: str, contexts: List[str]) -> str:
    separator = "\n\n----\n\n"
    context_block = separator.join(contexts)
    lang = detect_lang(question)
    if lang == "en":
        lang_rule = "Answer in English."
    elif lang == "zh":
        lang_rule = "Answer in Chinese."
    else:
        lang_rule = "If the question mixes English and Chinese, prefer Chinese in the answer."
    instructions = (
        "You are a helpful assistant. Answer the question using ONLY the context. "
        "If the answer is not in the context, say you don't know.\n"
        "Answer in the same language as the question. "
        f"{lang_rule}"
    )
    return f"{instructions}\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:"


def rag_answer(
    store: QdrantStore,
    settings: Settings,
    question: str,
    embed_model: str,
    chat_model: str,
    top_k: int,
) -> str:
    results = retrieve_top_k(
        store=store,
        collection=settings.default_collection,
        query=question,
        embed_model=embed_model,
        ollama_base_url=settings.ollama_base_url,
        top_k=top_k,
    )
    contexts = [r["text"] for r in results]
    prompt = build_prompt(question, contexts)
    return generate(prompt=prompt, model=chat_model, base_url=settings.ollama_base_url)

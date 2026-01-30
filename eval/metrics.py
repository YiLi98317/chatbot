from __future__ import annotations

from typing import Iterable, List, Sequence


def recall_at_k(truths: Sequence[str], predictions: Sequence[Sequence[str]], k: int = 10) -> float:
    """
    truths: list of gold labels (strings)
    predictions: list of prediction lists (strings), aligned with truths
    """
    if not truths or not predictions or len(truths) != len(predictions):
        return 0.0
    ok = 0
    total = len(truths)
    for t, preds in zip(truths, predictions):
        if any(p == t for p in preds[:k]):
            ok += 1
    return ok / max(1, total)


def mrr(truths: Sequence[str], predictions: Sequence[Sequence[str]]) -> float:
    if not truths or not predictions or len(truths) != len(predictions):
        return 0.0
    total = len(truths)
    acc = 0.0
    for t, preds in zip(truths, predictions):
        rr = 0.0
        for i, p in enumerate(preds, 1):
            if p == t:
                rr = 1.0 / i
                break
        acc += rr
    return acc / max(1, total)


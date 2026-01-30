from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Budgets:
    max_candidates_per_level: int
    max_rerank_inputs: int
    max_context_tokens: int
    per_stage_timeout_ms: int


def get_budgets() -> Budgets:
    return Budgets(
        max_candidates_per_level=int(os.getenv("MAX_CANDIDATES_PER_LEVEL", "100")),
        max_rerank_inputs=int(os.getenv("MAX_RERANK_INPUTS", "50")),
        max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "1200")),
        per_stage_timeout_ms=int(os.getenv("PER_STAGE_TIMEOUT_MS", "1500")),
    )


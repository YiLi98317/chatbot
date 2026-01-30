from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple


AGG_KEYWORDS = {
    "count",
    "sum",
    "avg",
    "average",
    "min",
    "max",
    "total",
}

JOIN_HINTS = {
    "join",
    "related to",
    "from album",
    "from artist",
    "tracks of",
    "tracks from",
    "by artist",
}


@dataclass(frozen=True)
class SQLIntent:
    has_aggregate: bool
    has_join: bool
    aggregate_funcs: List[str]
    join_hints: List[str]


def analyze_sql_intent(text: str) -> SQLIntent:
    s = (text or "").lower()
    aggs = sorted({w for w in AGG_KEYWORDS if w in s})
    joins = sorted({w for w in JOIN_HINTS if w in s})
    return SQLIntent(
        has_aggregate=bool(aggs),
        has_join=bool(joins),
        aggregate_funcs=aggs,
        join_hints=joins,
    )


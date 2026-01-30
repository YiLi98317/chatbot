from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .writer import write_trace_line


@dataclass
class LevelMetric:
    level: str
    latency_ms: float
    candidates_out: int
    stop_reason: str = ""
    cache_hit: bool = False
    error: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetric:
    query: str
    total_latency_ms: float
    levels: List[LevelMetric]
    lang: Optional[str] = None
    token_usage: Optional[int] = None
    error: Optional[str] = None


class MetricsRecorder:
    def __init__(self, traces_dir: str = "./traces"):
        self._traces_dir = Path(traces_dir)
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._levels: List[LevelMetric] = []
        self._t0: Optional[float] = None
        self._qtext: str = ""
        self._lang: Optional[str] = None

    def start(self, query_text: str, *, lang: Optional[str] = None) -> None:
        self._levels.clear()
        self._t0 = time.time()
        self._qtext = query_text or ""
        self._lang = lang

    def record_level(
        self,
        level: str,
        *,
        t_start: float,
        t_end: float,
        candidates_out: int,
        stop_reason: str = "",
        cache_hit: bool = False,
        error: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        latency_ms = max(0.0, (t_end - t_start) * 1000.0)
        self._levels.append(
            LevelMetric(
                level=level,
                latency_ms=latency_ms,
                candidates_out=int(candidates_out),
                stop_reason=stop_reason,
                cache_hit=bool(cache_hit),
                error=error,
                extras=extras or {},
            )
        )

    def end_and_write(self, *, token_usage: Optional[int] = None, error: Optional[str] = None) -> None:
        if self._t0 is None:
            return
        total_latency_ms = max(0.0, (time.time() - self._t0) * 1000.0)
        qm = QueryMetric(
            query=self._qtext,
            lang=self._lang,
            total_latency_ms=total_latency_ms,
            levels=list(self._levels),
            token_usage=token_usage,
            error=error,
        )
        write_trace_line(self._traces_dir, {"type": "metrics", "data": asdict(qm)})


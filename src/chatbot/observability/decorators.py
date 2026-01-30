from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from .metrics import MetricsRecorder


def timed_level(level_name: str, recorder: MetricsRecorder):
    """
    Decorator to measure a function call as a retrieval level and record metrics.
    The wrapped function must return a sequence (list-like) of candidates.
    """
    def _decorator(fn: Callable[..., Any]):
        @wraps(fn)
        def _wrapped(*args, **kwargs):
            t0 = time.time()
            error: Optional[str] = None
            try:
                result = fn(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                t1 = time.time()
                try:
                    candidates_out = 0
                    # best-effort length
                    out = locals().get("result", None)
                    if hasattr(out, "__len__"):
                        candidates_out = len(out)  # type: ignore[arg-type]
                except Exception:
                    candidates_out = 0
                recorder.record_level(
                    level=level_name,
                    t_start=t0,
                    t_end=t1,
                    candidates_out=candidates_out,
                    stop_reason="",
                    cache_hit=False,
                    error=error,
                    extras=None,
                )
        return _wrapped
    return _decorator


from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def write_trace_line(traces_dir: Path, payload: Dict[str, Any]) -> None:
    """
    Append a single JSON line to a daily metrics file in traces directory.
    """
    traces_dir.mkdir(parents=True, exist_ok=True)
    fname = traces_dir / f"metrics-{time.strftime('%Y%m%d')}.jsonl"
    with fname.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


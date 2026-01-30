#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def tail_metrics(traces_dir: str, n: int = 50) -> List[dict]:
    p = Path(traces_dir)
    if not p.exists():
        return []
    # Choose latest metrics file
    files = sorted([x for x in p.glob("metrics-*.jsonl") if x.is_file()])
    if not files:
        return []
    latest = files[-1]
    lines = latest.read_text(encoding="utf-8").strip().splitlines()[-n:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Basic ops checks for metrics traces.")
    ap.add_argument("--traces", default="./traces", help="Traces directory")
    ap.add_argument("--tail", type=int, default=50, help="Tail last N entries")
    args = ap.parse_args()
    rows = tail_metrics(args.traces, n=args.tail)
    latencies = []
    for r in rows:
        try:
            if r.get("type") == "metrics":
                latencies.append(float(r["data"]["total_latency_ms"]))
        except Exception:
            continue
    if not latencies:
        print("No metrics entries found.")
        return
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[int(0.50 * (len(latencies_sorted) - 1))]
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
    print(f"count={len(latencies_sorted)} p50_ms={p50:.1f} p95_ms={p95:.1f}")


if __name__ == "__main__":
    main()


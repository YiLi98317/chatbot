Overview
========

This minimal dashboard is based on JSON lines written to ./traces/metrics-YYYYMMDD.jsonl.

Each line has the shape:

{
  "type": "metrics",
  "data": {
    "query": "...",
    "total_latency_ms": 123.4,
    "levels": [
      {"level": "L1_BM25", "latency_ms": 12.3, "candidates_out": 10, "stop_reason": "", "cache_hit": false, "error": null, "extras": {}}
    ],
    "token_usage": null,
    "error": null
  }
}

Quickstart
----------

1) Run queries in debug/dev to produce traces.
2) Inspect recent file:

  tail -n 50 ./traces/metrics-$(date +%Y%m%d).jsonl

3) Pipe to jq for simple summaries (if installed):

  jq -r '.data.total_latency_ms' ./traces/metrics-$(date +%Y%m%d).jsonl

4) Build basic p95:

  awk -F: '/total_latency_ms/ {gsub(/[ ,]/,"",$2); print $2}' ./traces/metrics-$(date +%Y%m%d).jsonl | sort -n | awk ' { a[i++]=$1; } END { print a[int(i*0.95)] } '


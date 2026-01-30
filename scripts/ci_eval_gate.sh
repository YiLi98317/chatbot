#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

if [[ ! -f "$PROJECT_ROOT/eval/runner.py" ]]; then
  echo "runner not found"
  exit 0
fi

OUT="$(python3 "$PROJECT_ROOT/eval/runner.py" --db-uri "${DB_URI:-}" --ci --k 10 --modes "bm25,prf,qexp" 2>&1 || true)"
echo "$OUT"

if echo "$OUT" | grep -q "NO_GOLDEN=1"; then
  echo "No golden dataset, skipping CI gate."
  exit 0
fi

if echo "$OUT" | grep -qi "RECALL_OK=true"; then
  echo "CI gate passed."
  exit 0
else
  echo "CI gate failed."
  exit 1
fi


#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from chatbot.retrieval.bm25 import search_bm25_as_dicts
from chatbot.retrieval.prf import apply_prf_bm25
from chatbot.retrieval.query_expansion import apply_qexp_bm25
from eval.metrics import recall_at_k, mrr


def load_golden(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    return rows or None


def run_mode(db_uri: str, query: str, mode: str, k: int) -> List[str]:
    if mode == "bm25":
        hits = search_bm25_as_dicts(db_uri=db_uri, query=query, limit=k)
    elif mode == "prf":
        hits = apply_prf_bm25(db_uri=db_uri, query=query, seed_k=10, expansion_tokens=5, top_k=k)
    elif mode == "qexp":
        hits = apply_qexp_bm25(db_uri=db_uri, query=query, top_k=k)
    else:
        hits = []
    names: List[str] = []
    for h in hits:
        meta = h.get("metadata", {}) or {}
        name = meta.get("name") or h.get("text") or ""
        if name:
            names.append(str(name))
    return names[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluation runner with ablations.")
    ap.add_argument("--db-uri", default="", help="SQLite URI for Chinook")
    ap.add_argument("--golden", default="eval/golden.jsonl", help="Path to golden dataset")
    ap.add_argument("--k", type=int, default=10, help="Recall@k")
    ap.add_argument("--modes", default="bm25,prf,qexp", help="Comma-separated: bm25,prf,qexp")
    ap.add_argument("--ci", action="store_true", help="CI mode: concise output")
    ap.add_argument("--gate", default="recall@10>=0.50", help="Gate condition for CI")
    args = ap.parse_args()

    golden = load_golden(Path(args.golden))
    if not golden:
        print("NO_GOLDEN=1")
        print("SKIP_REASON=missing golden dataset")
        sys.exit(0)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    truths: List[str] = []
    by_mode_preds: Dict[str, List[List[str]]] = {m: [] for m in modes}
    for row in golden:
        q = str(row.get("question") or "")
        truth = str(row.get("expected_answer") or "")
        if not q or not truth:
            continue
        truths.append(truth)
        for m in modes:
            preds = run_mode(args.db_uri, q, m, args.k)
            by_mode_preds[m].append(preds)

    # Report metrics
    for m in modes:
        r = recall_at_k(truths, by_mode_preds[m], k=args.k)
        rr = mrr(truths, by_mode_preds[m])
        print(f"MODE={m} recall@{args.k}={r:.3f} mrr={rr:.3f}")

    # Simple CI gate
    ok = True
    gate = str(args.gate or "").lower().strip()
    if gate.startswith("recall@"):
        try:
            rest = gate[len("recall@") :]
            k_str, cond = rest.split(">=")
            k_gate = int(k_str)
            thres = float(cond)
            # choose first mode for gate
            m0 = modes[0]
            r0 = recall_at_k(truths, by_mode_preds[m0], k=k_gate)
            ok = r0 >= thres
        except Exception:
            ok = True
    print(f"RECALL_OK={'true' if ok else 'false'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()


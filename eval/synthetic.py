#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    """
    Write a tiny synthetic golden file if none exists.
    This is a placeholder; replace with real golden examples.
    """
    out = Path("eval/golden.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print("golden exists, not overwriting:", out)
        return
    rows = [
        {"id": "ex1", "question": "song called 'Black Dog'", "expected_answer": "Black Dog"},
        {"id": "ex2", "question": "track named 'Asche zu Asche'", "expected_answer": "Asche zu Asche"},
    ]
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote synthetic golden:", out)


if __name__ == "__main__":
    main()


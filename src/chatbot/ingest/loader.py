from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from chatbot.ingest.chunking import chunk_text_en, chunk_text_zh
from chatbot.retrieval.normalize import detect_lang


def iter_files(root: str, extensions: Tuple[str, ...] = (".txt", ".md")) -> Iterable[Path]:
    base = Path(root)
    for path in base.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_and_chunk(root: str, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    docs: List[Dict] = []
    for path in iter_files(root):
        raw = read_text(path)
        lang = detect_lang(raw)

        # If user didn't override defaults, allow zh-specific tuning via env vars.
        # (Heuristic: CLI defaults are 800/150; if those are used, apply ZH_*.)
        zh_chunk_size = int(os.getenv("ZH_CHUNK_SIZE", str(chunk_size)))
        zh_chunk_overlap = int(os.getenv("ZH_CHUNK_OVERLAP", str(chunk_overlap)))
        eff_size = chunk_size
        eff_overlap = chunk_overlap
        if lang in ("zh", "mixed") and chunk_size == 800:
            eff_size = zh_chunk_size
        if lang in ("zh", "mixed") and chunk_overlap == 150:
            eff_overlap = zh_chunk_overlap

        if lang in ("zh", "mixed"):
            chunks = chunk_text_zh(raw, max_chars=eff_size, overlap=eff_overlap)
        else:
            chunks = chunk_text_en(raw, max_chars=eff_size, overlap=eff_overlap)

        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": str(path),
                        "chunk": i,
                        "lang": lang,
                        "source_type": "file",
                    },
                }
            )
    return docs

from __future__ import annotations
import hashlib
from typing import Dict, Mapping
import uuid
import re
from typing import Mapping, Any


def doc_id(table: str, pk_value: object, updated_at_value: object) -> str:
    raw = f"{table}:{pk_value}:{updated_at_value}"
    # Deterministic UUID v5 to meet Qdrant ID requirements (UUID or uint)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


EXACT_EXCLUDE = {
    "updated_at",
    "bytes",
    "milliseconds",
    "postalcode",
    "phone",
    "fax",
    "address",
    "state",
    "unitprice",
}
ID_KEY_RE = re.compile(r".*id$", re.IGNORECASE)


def row_to_text(table: str, row: Mapping[str, Any]) -> str:
    lines = []

    for k, v in row.items():
        if v is None:
            continue

        lk = k.lower()

        # Drop noisy fields
        if lk in EXACT_EXCLUDE:
            continue
        if ID_KEY_RE.match(k) and lk not in {
            "trackid"
        }:  # keep none of the IDs in text; adjust as desired
            continue

        # Drop very long / very numeric strings (common in addresses, etc.)
        sv = str(v).strip()
        if len(sv) > 200:
            continue

        lines.append(f"{k}: {sv}")

    return "\n".join(lines)

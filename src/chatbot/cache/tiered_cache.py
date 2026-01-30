from __future__ import annotations

import collections
from typing import Any, Dict, Optional, Tuple


class LRUCache:
    def __init__(self, capacity: int = 256):
        self._cap = max(1, int(capacity))
        self._store: "collections.OrderedDict[str, Any]" = collections.OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        val = self._store.pop(key)
        self._store[key] = val
        return val

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self._cap:
            self._store.popitem(last=False)
        self._store[key] = value


class TieredCache:
    """
    Simple two-tier in-memory cache (hot LRU + warm LRU).
    """
    def __init__(self, hot_capacity: int = 256, warm_capacity: int = 1024):
        self.hot = LRUCache(hot_capacity)
        self.warm = LRUCache(warm_capacity)

    def get(self, key: str) -> Optional[Any]:
        v = self.hot.get(key)
        if v is not None:
            return v
        v = self.warm.get(key)
        if v is not None:
            # promote to hot
            self.hot.set(key, v)
        return v

    def set(self, key: str, value: Any, *, warm: bool = False) -> None:
        (self.warm if warm else self.hot).set(key, value)


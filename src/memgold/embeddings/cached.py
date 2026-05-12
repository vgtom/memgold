"""Embedding cache wrapper (bytes cache + in-flight dedupe)."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from memgold.interfaces.embedder import Embedder

if TYPE_CHECKING:
    from memgold.interfaces.storage import CacheStore


class CachedEmbedder(Embedder):
    """Caches embeddings by sha256(text) using a :class:`CacheStore`.

    TODO: Add cross-process cache (Redis) and negative-cache error handling.
    """

    def __init__(self, inner: Embedder, cache: "CacheStore", *, key_prefix: str = "emb:") -> None:
        self._inner = inner
        self._cache = cache
        self._prefix = key_prefix
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def _key(self, text: str) -> str:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._prefix}{h}"

    def _encode(self, vec: list[float]) -> bytes:
        return ",".join(f"{v:.8f}" for v in vec).encode("utf-8")

    def _decode(self, raw: bytes) -> list[float]:
        return [float(x) for x in raw.decode("utf-8").split(",") if x]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            key = self._key(text)
            cached = await self._cache.get(key)
            if cached is not None:
                out.append(self._decode(cached))
                continue
            lock = self._locks.setdefault(key, asyncio.Lock())
            async with lock:
                cached2 = await self._cache.get(key)
                if cached2 is not None:
                    out.append(self._decode(cached2))
                    continue
                (vec,) = await self._inner.embed_texts([text])
                await self._cache.set(key, self._encode(vec), ttl_seconds=86400 * 7)
                out.append(vec)
        return out

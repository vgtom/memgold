"""Application service for batched async embedding."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.embedder import Embedder


class EmbeddingService:
    """Batching wrapper over an :class:`Embedder` with simple concurrency control.

    TODO: Token-aware batching for provider limits; circuit breaker on failures.
    """

    def __init__(self, embedder: "Embedder", settings: "MemoryLayerSettings") -> None:
        self._embedder = embedder
        self._settings = settings
        self._sem = asyncio.Semaphore(4)

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in chunks of ``embedding_batch_size``."""
        if not texts:
            return []
        batch = max(1, self._settings.embedding_batch_size)
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            async with self._sem:
                out.extend(await self._embedder.embed_texts(chunk))
        return out

    async def embed_query(self, text: str) -> list[float]:
        """Single-query convenience."""
        (vec,) = await self.embed_texts([text])
        return vec

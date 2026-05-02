"""Persist new memories from user-provided text."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memgold.extraction.extractor import MemoryExtractor
from memgold.models.memory import Memory
from memgold.storage.memory_stores import memory_to_json

if TYPE_CHECKING:
    from memgold.storage.base import KVStore, VectorStore


class MemoryWriteService:
    """Coordinates extraction and dual writes (vector + KV)."""

    def __init__(
        self,
        extractor: MemoryExtractor,
        vector_store: "VectorStore",
        kv_store: "KVStore",
    ) -> None:
        self._extractor = extractor
        self._vector_store = vector_store
        self._kv_store = kv_store

    async def process_input(self, user_id: str, text: str) -> list[Memory]:
        """Extract memories from *text* and persist them for *user_id*."""
        raw_memories = await self._extractor.extract(text)
        stored: list[Memory] = []
        for item in raw_memories:
            memory = item.model_copy(update={"user_id": user_id})
            await self._vector_store.add(memory)
            await self._kv_store.set(str(memory.id), memory_to_json(memory))
            stored.append(memory)
        return stored

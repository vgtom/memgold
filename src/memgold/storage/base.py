"""Abstract storage interfaces for vector and key-value backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memgold.models.memory import Memory


class VectorStore(ABC):
    """Embedding-addressable memory index."""

    @abstractmethod
    async def add(self, memory: "Memory") -> None:
        """Persist *memory* (implementations compute or attach embeddings as needed)."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        user_id: str,
    ) -> list["Memory"]:
        """Return up to *top_k* memories for *user_id* ranked by similarity to *query_embedding*."""


class KVStore(ABC):
    """Opaque key-value sidecar (full rows, JSON blobs, cache metadata, etc.)."""

    @abstractmethod
    async def set(self, key: str, value: str) -> None:
        """Associate *value* with *key* (replace if exists)."""

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """Return the value for *key*, or ``None`` if missing."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key* if present."""

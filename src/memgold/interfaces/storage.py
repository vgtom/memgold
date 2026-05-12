"""Storage ports: vector, structured memory, graph, and cache."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from memgold.models.enums import MemoryType
    from memgold.models.graph import MemoryEdge
    from memgold.models.memory import Memory


class VectorStore(ABC):
    """Embedding-addressable index with optional scoped filters."""

    @abstractmethod
    async def upsert(self, memory: "Memory", embedding: list[float]) -> None:
        """Index *memory* with the supplied dense *embedding*."""

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """Remove vectors for *memory_id* if present."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        top_k: int,
        hierarchy_prefix: str | None = None,
        memory_types: frozenset["MemoryType"] | None = None,
        session_id: str | None = None,
    ) -> list["Memory"]:
        """Vector similarity search with optional metadata filters."""


class MemoryStore(ABC):
    """Authoritative metadata and text for memories."""

    @abstractmethod
    async def upsert(self, memory: "Memory") -> None:
        """Insert or replace *memory* by id."""

    @abstractmethod
    async def get(self, memory_id: UUID) -> "Memory | None":
        """Fetch a memory by id."""

    @abstractmethod
    async def delete(self, memory_id: UUID) -> None:
        """Delete a memory row."""

    @abstractmethod
    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 10_000,
        include_archived: bool = False,
    ) -> list["Memory"]:
        """Return memories owned by *user_id*."""

    @abstractmethod
    async def keyword_search(
        self,
        user_id: str,
        query: str,
        *,
        limit: int,
    ) -> list["Memory"]:
        """Symbolic / lexical retrieval (BM25-like scoring optional in adapters)."""

    @abstractmethod
    async def hierarchy_scope(
        self,
        user_id: str,
        path_prefix: str,
        *,
        limit: int,
    ) -> list["Memory"]:
        """Memories whose ``hierarchy_path`` starts with *path_prefix*."""


class GraphStore(ABC):
    """Lightweight adjacency persistence for memory linking."""

    @abstractmethod
    async def add_edge(self, edge: "MemoryEdge") -> None:
        """Persist a directed edge."""

    @abstractmethod
    async def remove_edge(self, edge_id: UUID) -> None:
        """Delete edge by id."""

    @abstractmethod
    async def neighbors(
        self,
        memory_id: UUID,
        *,
        direction: str = "both",
        limit: int = 64,
    ) -> list[UUID]:
        """Return adjacent memory ids (string targets coerced by caller)."""


class CacheStore(ABC):
    """Ephemeral KV cache (session buffers, embed cache, rate limits)."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None:
        """Return raw bytes or ``None``."""

    @abstractmethod
    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        """Associate *value* with *key* with optional TTL."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key* if present."""

"""Retrieval request/response DTOs."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from memgold.models.enums import MemoryType


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Scoped hybrid retrieval parameters."""

    user_id: str
    query: str
    top_k: int = 12
    hierarchy_prefix: str | None = None
    memory_types: frozenset[MemoryType] | None = None
    session_id: str | None = None
    expand_graph: bool = True
    graph_neighbor_limit: int = 24


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """A memory with fused retrieval diagnostics."""

    memory_id: UUID
    fused_score: float
    components: dict[str, float] = field(default_factory=dict)

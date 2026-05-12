"""Read path: retrieval service adapter for legacy DI names."""

from __future__ import annotations

from memgold.models.enums import MemoryType
from memgold.models.memory import Memory
from memgold.retrieval.types import RetrievalRequest
from memgold.services.retrieval_service import RetrievalService


class MemoryReadService:
    """Thin adapter mapping simple (user, query) calls to :class:`RetrievalService`."""

    def __init__(self, retrieval: RetrievalService) -> None:
        self._retrieval = retrieval

    async def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 10,
        hierarchy_prefix: str | None = None,
        memory_types: list[MemoryType] | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        """Return ranked memories relevant to *query* for *user_id*."""
        types = frozenset(memory_types) if memory_types else None
        req = RetrievalRequest(
            user_id=user_id,
            query=query,
            top_k=top_k,
            hierarchy_prefix=hierarchy_prefix,
            memory_types=types,
            session_id=session_id,
        )
        return await self._retrieval.search(req)

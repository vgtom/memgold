"""High-level retrieval with ranking and access accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from memgold.models.memory import Memory
from memgold.retrieval.hybrid import HybridRetriever
from memgold.retrieval.types import RetrievalRequest
from memgold.services.ranking_service import RankingService

if TYPE_CHECKING:
    from memgold.interfaces.storage import MemoryStore


class RetrievalService:
    """Hybrid retrieval + rerank + optional memory touches for salience feedback."""

    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        ranking: RankingService,
        memory_store: "MemoryStore",
    ) -> None:
        self._retriever = retriever
        self._ranking = ranking
        self._memory_store = memory_store

    async def search(self, request: RetrievalRequest, *, record_access: bool = True) -> list[Memory]:
        """Return ranked memories for *request*."""
        scored = await self._retriever.retrieve(request)
        ranked = self._ranking.rank(scored)
        if record_access:
            for m in ranked[: min(8, len(ranked))]:
                m.touch_access()
                await self._memory_store.upsert(m)
        return ranked

    async def get(self, memory_id: UUID) -> Memory | None:
        """Fetch a single memory and bump access stats."""
        mem = await self._memory_store.get(memory_id)
        if mem is None:
            return None
        mem.touch_access()
        await self._memory_store.upsert(mem)
        return mem

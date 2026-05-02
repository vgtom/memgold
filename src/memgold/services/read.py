"""Read path: retrieve candidate memories and re-rank."""

from __future__ import annotations

from memgold.models.memory import Memory
from memgold.ranking.ranker import MemoryRanker
from memgold.retrieval.retriever import MemoryRetriever


class MemoryReadService:
    """Hybrid retrieval followed by ranking."""

    def __init__(self, retriever: MemoryRetriever, ranker: MemoryRanker) -> None:
        self._retriever = retriever
        self._ranker = ranker

    async def retrieve(self, user_id: str, query: str) -> list[Memory]:
        """Return ranked memories relevant to *query* for *user_id*."""
        candidates = await self._retriever.retrieve(query, user_id)
        return await self._ranker.rank(candidates)

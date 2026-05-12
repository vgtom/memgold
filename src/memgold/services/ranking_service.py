"""Facade over :class:`~memgold.ranking.pipeline.RankingPipeline` for DI clarity."""

from __future__ import annotations

from memgold.models.memory import Memory
from memgold.ranking.pipeline import RankingPipeline
from memgold.retrieval.types import ScoredMemory


class RankingService:
    """Thin wrapper so ranking can be swapped or decorated independently."""

    def __init__(self, pipeline: RankingPipeline) -> None:
        self._pipeline = pipeline

    def rank(self, scored: list[tuple[Memory, ScoredMemory]]) -> list[Memory]:
        """Delegate to underlying pipeline."""
        return self._pipeline.rank(scored)

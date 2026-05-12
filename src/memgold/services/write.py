"""Persist new memories from user-provided text (legacy facade)."""

from __future__ import annotations

from memgold.models.memory import Memory
from memgold.services.ingestion import MemoryIngestionService


class MemoryWriteService:
    """Coordinates extraction and dual writes via :class:`MemoryIngestionService`."""

    def __init__(self, ingestion: MemoryIngestionService) -> None:
        self._ingestion = ingestion

    async def process_input(self, user_id: str, text: str) -> list[Memory]:
        """Extract memories from *text* and persist them for *user_id*."""
        return await self._ingestion.ingest_text(user_id=user_id, text=text)

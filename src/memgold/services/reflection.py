"""Reflection jobs that synthesize higher-level memories from recent activity."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from memgold.models.enums import MemorySource, MemoryType
from memgold.models.memory import Memory
from memgold.observability.logging import log_event

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.storage import MemoryStore, VectorStore
    from memgold.services.embedding_service import EmbeddingService


class ReflectionService:
    """Produces reflective memories from recent episodic/semantic signals."""

    def __init__(
        self,
        *,
        memory_store: "MemoryStore",
        vector_store: "VectorStore",
        embedding_service: "EmbeddingService",
        settings: "MemoryLayerSettings",
    ) -> None:
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._embed = embedding_service
        self._settings = settings

    async def reflect(self, *, user_id: str, lookback: int = 24) -> Memory | None:
        """Scan recent memories and persist a single reflective summary.

        TODO: LLM-based reflective writing; multi-step self-critique; safety filters.
        """
        mems = await self._memory_store.list_for_user(user_id, limit=lookback, include_archived=False)
        if not mems:
            return None
        mems.sort(key=lambda m: m.created_at, reverse=True)
        window = mems[:lookback]
        themes = ", ".join(sorted({m.semantic_cluster or "general" for m in window if m.semantic_cluster}))
        text = (
            f"Reflection: recurring themes={themes or 'n/a'}; "
            f"sample={window[0].content[:160]!r}"
        )
        out = Memory(
            id=uuid4(),
            user_id=user_id,
            content=text,
            summary="Automated reflection",
            memory_type=MemoryType.PROCEDURAL,
            hierarchy_path=window[0].hierarchy_path,
            confidence=0.45,
            salience_score=0.5,
            source=MemorySource.REFLECTION,
            metadata={"basis_ids": [str(m.id) for m in window[:8]]},
        )
        await self._memory_store.upsert(out)
        (emb,) = await self._embed.embed_texts([out.content])
        await self._vector_store.upsert(out, emb)
        log_event("memory_reflect", {"user_id": user_id}, settings=self._settings)
        return out

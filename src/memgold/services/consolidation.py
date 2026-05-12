"""Memory consolidation jobs (summaries, merges, room rollups)."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from memgold.models.enums import MemorySource, MemoryType
from memgold.models.memory import Memory, normalize_hierarchy_path
from memgold.observability.logging import log_event

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.storage import MemoryStore, VectorStore
    from memgold.services.embedding_service import EmbeddingService


class ConsolidationService:
    """Batch workflows that compress redundant episodic traces and summarize rooms."""

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

    def _room_key(self, path: str) -> str:
        parts = [p for p in normalize_hierarchy_path(path).split("/") if p]
        return "/" + "/".join(parts[:2]) if len(parts) >= 2 else normalize_hierarchy_path(path)

    async def consolidate_user(self, *, user_id: str) -> list[Memory]:
        """Cluster episodic memories by palace room and emit compact summaries.

        TODO: LLM summarization; dedupe via embeddings; transactional commits.
        """
        mems = await self._memory_store.list_for_user(user_id, limit=5000, include_archived=False)
        episodic = [m for m in mems if m.memory_type == MemoryType.EPISODIC]
        groups: dict[str, list[Memory]] = defaultdict(list)
        for m in episodic:
            groups[self._room_key(m.hierarchy_path)].append(m)
        created: list[Memory] = []
        for room, items in groups.items():
            if len(items) < 4:
                continue
            items.sort(key=lambda m: m.created_at)
            tail = items[-12:]
            summary = " | ".join(t.content[:120] for t in tail)
            mem = Memory(
                id=uuid4(),
                user_id=user_id,
                content=f"Room summary for {room}: {summary}",
                summary=f"Consolidated {len(tail)} episodic items",
                memory_type=MemoryType.SEMANTIC,
                hierarchy_path=normalize_hierarchy_path(room + "/_meta/summary"),
                salience_score=0.55,
                confidence=0.5,
                source=MemorySource.CONSOLIDATION,
                metadata={"consolidated_ids": [str(t.id) for t in tail]},
            )
            await self._memory_store.upsert(mem)
            (emb,) = await self._embed.embed_texts([mem.content])
            await self._vector_store.upsert(mem, emb)
            created.append(mem)
        log_event(
            "memory_consolidate",
            {"user_id": user_id, "summaries": len(created)},
            settings=self._settings,
        )
        return created

    async def summarize_selection(self, *, user_id: str, memory_ids: list[UUID]) -> Memory | None:
        """Merge selected memories into a single summarized semantic record.

        TODO: Replace concatenation with LLM summarization and provenance tracking.
        """
        mems: list[Memory] = []
        for mid in memory_ids:
            m = await self._memory_store.get(mid)
            if m is not None and m.user_id == user_id and not m.archived:
                mems.append(m)
        if not mems:
            return None
        merged = "\n".join(m.content for m in mems[:32])
        path = self._room_key(mems[0].hierarchy_path)
        mem = Memory(
            id=uuid4(),
            user_id=user_id,
            content=f"Summary bundle:\n{merged[:8000]}",
            summary=f"Merged {len(mems)} memories",
            memory_type=MemoryType.SEMANTIC,
            hierarchy_path=normalize_hierarchy_path(path + "/_meta/bundle"),
            salience_score=0.52,
            confidence=0.48,
            source=MemorySource.CONSOLIDATION,
            metadata={"summarized_ids": [str(m.id) for m in mems]},
        )
        await self._memory_store.upsert(mem)
        (emb,) = await self._embed.embed_texts([mem.content])
        await self._vector_store.upsert(mem, emb)
        log_event(
            "memory_summarize",
            {"user_id": user_id, "sources": len(mems)},
            settings=self._settings,
        )
        return mem

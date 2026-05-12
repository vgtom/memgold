"""End-to-end memory ingestion orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from memgold.extraction.contradiction import detect_contradictions, edge_from_suggestion, links_from_contradictions
from memgold.extraction.importance import score_importance
from memgold.models.enums import GraphRelationType
from memgold.models.events import IngestionEvent
from memgold.models.graph import MemoryEdge
from memgold.models.memory import Memory
from memgold.observability.logging import log_event

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.extraction import ExtractionPipeline
    from memgold.interfaces.storage import GraphStore, MemoryStore, VectorStore
    from memgold.models.extraction import MemoryCandidate
    from memgold.services.embedding_service import EmbeddingService
    from memgold.services.working_memory import WorkingMemoryBuffer


class MemoryIngestionService:
    """Coordinates extraction, scoring, dual writes, and graph mutations."""

    def __init__(
        self,
        *,
        pipeline: "ExtractionPipeline",
        memory_store: "MemoryStore",
        vector_store: "VectorStore",
        graph_store: "GraphStore",
        embedding_service: "EmbeddingService",
        settings: "MemoryLayerSettings",
        working_memory: "WorkingMemoryBuffer | None" = None,
    ) -> None:
        self._pipeline = pipeline
        self._memory_store = memory_store
        self._vector_store = vector_store
        self._graph_store = graph_store
        self._embed = embedding_service
        self._settings = settings
        self._wm = working_memory

    def _candidate_to_memory(
        self,
        candidate: "MemoryCandidate",
        *,
        user_id: str,
        session_id: str | None,
        text_len: int,
    ) -> Memory:
        salience = score_importance(candidate, text_len=text_len)
        return Memory(
            id=uuid4(),
            user_id=user_id,
            session_id=session_id,
            content=candidate.content,
            summary=candidate.summary,
            memory_type=candidate.memory_type,
            hierarchy_path=candidate.hierarchy_path,
            semantic_cluster=candidate.semantic_cluster,
            entities=list(candidate.entities),
            keywords=list(candidate.keywords),
            confidence=float(candidate.confidence),
            salience_score=float(max(candidate.salience_score, salience)),
            source=candidate.source,
            metadata=dict(candidate.metadata),
        )

    async def ingest_text(
        self,
        *,
        user_id: str,
        text: str,
        session_id: str | None = None,
        role: str = "user",
    ) -> list[Memory]:
        """Run extraction, persist memories, index vectors, and wire contradiction edges."""
        if self._wm is not None and session_id:
            await self._wm.append_message(user_id=user_id, session_id=session_id, role=role, text=text)
        candidates = await self._pipeline.build_candidates(
            text,
            user_id=user_id,
            session_id=session_id,
        )
        existing = await self._memory_store.list_for_user(user_id, limit=400, include_archived=False)
        stored: list[Memory] = []
        texts_for_embed: list[str] = []
        memories_for_embed: list[Memory] = []

        for cand in candidates:
            mem = self._candidate_to_memory(cand, user_id=user_id, session_id=session_id, text_len=len(text))
            signals = detect_contradictions(mem.content, existing + stored)
            for s in signals:
                mem.metadata.setdefault("contradiction_signals", []).append(
                    {
                        "existing_id": str(s.existing_memory_id),
                        "score": s.score,
                    },
                )
            await self._memory_store.upsert(mem)
            texts_for_embed.append(mem.content)
            memories_for_embed.append(mem)
            for link in links_from_contradictions(mem.id, signals, threshold=0.55):
                await self._graph_store.add_edge(edge_from_suggestion(link))
            # soft relate to episodic root of batch
            if stored and mem.memory_type.name != "EPISODIC":
                await self._graph_store.add_edge(
                    MemoryEdge(
                        source_id=mem.id,
                        target_id=stored[0].id,
                        relation=GraphRelationType.RELATED_TO,
                        weight=0.35,
                    ),
                )
            stored.append(mem)
            existing.append(mem)

        if texts_for_embed:
            embeddings = await self._embed.embed_texts(texts_for_embed)
            for mem, emb in zip(memories_for_embed, embeddings, strict=True):
                await self._vector_store.upsert(mem, emb)

        log_event(
            "memory_ingest",
            {"user_id": user_id, "count": len(stored), "session_id": session_id},
            settings=self._settings,
        )
        return stored

    async def ingest_event(self, event: IngestionEvent) -> list[Memory]:
        """Ingest a structured :class:`~memgold.models.events.IngestionEvent`.

        TODO: Materialize ``DERIVED_FROM`` edges to an ``EventStore`` once event
        nodes are first-class in :class:`~memgold.interfaces.storage.GraphStore`.
        """
        mems = await self.ingest_text(
            user_id=event.user_id,
            text=event.text,
            session_id=event.session_id,
            role=event.role,
        )
        eid = str(event.id)
        for m in mems:
            m.metadata["ingestion_event_id"] = eid
            m.metadata.setdefault("event_metadata", {}).update(event.metadata)
            await self._memory_store.upsert(m)
        return mems

"""Hybrid retrieval: dense + lexical + hierarchy + graph expansion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from memgold.models.memory import Memory, normalize_hierarchy_path
from memgold.retrieval.types import RetrievalRequest, ScoredMemory

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.storage import GraphStore, MemoryStore, VectorStore
    from memgold.services.embedding_service import EmbeddingService


def _recency_weight(created_at: datetime, *, max_age_seconds: float) -> float:
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    age = max(0.0, (now - created_at).total_seconds())
    return max(0.0, 1.0 - min(age / max_age_seconds, 1.0))


def _keyword_overlap_score(query: str, memory: Memory) -> float:
    from memgold.storage.in_memory import _keyword_score  # reuse token overlap

    return _keyword_score(query, memory)


class HybridRetriever:
    """Fuses vector, keyword, hierarchy scope, recency, and optional graph hops."""

    def __init__(
        self,
        *,
        vector_store: "VectorStore",
        memory_store: "MemoryStore",
        graph_store: "GraphStore",
        embedding_service: "EmbeddingService",
        settings: "MemoryLayerSettings",
    ) -> None:
        self._vector = vector_store
        self._memory = memory_store
        self._graph = graph_store
        self._embed = embedding_service
        self._settings = settings

    async def retrieve(self, request: RetrievalRequest) -> list[tuple[Memory, ScoredMemory]]:
        """Return memories with score envelopes sorted by ``fused_score`` descending."""
        q_emb = await self._embed.embed_query(request.query)
        vec_hits = await self._vector.search(
            q_emb,
            user_id=request.user_id,
            top_k=max(request.top_k * 3, request.top_k),
            hierarchy_prefix=request.hierarchy_prefix,
            memory_types=request.memory_types,
            session_id=request.session_id,
        )
        kw_hits = await self._memory.keyword_search(
            request.user_id,
            request.query,
            limit=max(request.top_k * 3, request.top_k),
        )

        prefix = normalize_hierarchy_path(request.hierarchy_prefix) if request.hierarchy_prefix else None
        hier_hits: list[Memory] = []
        if prefix:
            hier_hits = await self._memory.hierarchy_scope(
                request.user_id,
                prefix,
                limit=max(request.top_k * 2, request.top_k),
            )

        by_id: dict[UUID, Memory] = {}
        vec_score: dict[UUID, float] = {}
        kw_score: dict[UUID, float] = {}
        hier_score: dict[UUID, float] = {}

        for rank, m in enumerate(vec_hits):
            by_id[m.id] = m
            vec_score[m.id] = max(0.0, 1.0 - rank / max(len(vec_hits), 1))
        for rank, m in enumerate(kw_hits):
            by_id.setdefault(m.id, m)
            kw_score[m.id] = max(kw_score.get(m.id, 0.0), _keyword_overlap_score(request.query, m))
        for m in hier_hits:
            by_id.setdefault(m.id, m)
            hier_score[m.id] = max(hier_score.get(m.id, 0.0), 0.35)

        if request.expand_graph and self._settings.enable_graph_expansion:
            seeds = list(by_id.keys())[: request.top_k]
            hop_limit = max(1, self._settings.graph_expansion_hops)
            frontier = list(seeds)
            seen: set[UUID] = set(seeds)
            for _ in range(hop_limit):
                next_ids: list[UUID] = []
                for mid in frontier:
                    nbrs = await self._graph.neighbors(mid, limit=request.graph_neighbor_limit)
                    for n in nbrs:
                        if n in seen:
                            continue
                        seen.add(n)
                        next_ids.append(n)
                frontier = next_ids
                for nid in next_ids:
                    mem = await self._memory.get(nid)
                    if mem is None or mem.archived:
                        continue
                    if mem.user_id != request.user_id:
                        continue
                    by_id.setdefault(mem.id, mem)
                    hier_score.setdefault(mem.id, 0.0)
                    vec_score.setdefault(mem.id, 0.0)
                    kw_score.setdefault(mem.id, 0.0)
                    # graph prior
                    vec_score[mem.id] = max(vec_score[mem.id], 0.15)

        max_age = 365.0 * 24 * 3600
        fused: list[tuple[Memory, ScoredMemory]] = []
        for mid, mem in by_id.items():
            if request.memory_types is not None and mem.memory_type not in request.memory_types:
                continue
            if request.session_id is not None:
                if mem.session_id is not None and mem.session_id != request.session_id:
                    continue
            v = vec_score.get(mid, 0.0)
            k = kw_score.get(mid, 0.0)
            h = hier_score.get(mid, 0.0)
            r = _recency_weight(mem.created_at, max_age_seconds=max_age)
            s = float(mem.salience_score)
            d = float(mem.decay_score)
            # memory type prior
            mt = 1.0
            if mem.memory_type.name == "EPISODIC":
                mt = 1.05
            elif mem.memory_type.name == "PROCEDURAL":
                mt = 1.02
            fused_val = (
                0.45 * v
                + 0.25 * k
                + 0.1 * h
                + 0.12 * r
                + 0.05 * s
            ) * (0.85 + 0.15 * d) * mt
            fused_val = float(max(0.0, min(1.5, fused_val)))
            fused.append(
                (
                    mem,
                    ScoredMemory(
                        memory_id=mid,
                        fused_score=fused_val,
                        components={
                            "vector": v,
                            "keyword": k,
                            "hierarchy": h,
                            "recency": r,
                            "salience": s,
                            "decay": d,
                            "type_prior": mt,
                        },
                    ),
                ),
            )
        fused.sort(key=lambda t: t[1].fused_score, reverse=True)
        return fused[: request.top_k]


class SemanticExpander:
    """Optional query expansion hook (stub lists synonyms as TODO)."""

    @staticmethod
    async def expand_query(query: str) -> str:
        """Return expanded query string.

        TODO: LLM synonym expansion / pseudo-relevance feedback from top hits.
        """
        return query

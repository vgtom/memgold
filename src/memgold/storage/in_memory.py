"""In-memory storage adapters for development, tests, and local runs."""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from uuid import UUID

from memgold.interfaces.storage import CacheStore, GraphStore, MemoryStore, VectorStore
from memgold.models.enums import MemoryType
from memgold.models.graph import MemoryEdge
from memgold.models.memory import Memory, normalize_hierarchy_path


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _keyword_score(query: str, memory: Memory) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    corpus = _tokens(memory.content) | _tokens(memory.summary or "") | {k.lower() for k in memory.keywords}
    if not corpus:
        return 0.0
    overlap = len(q & corpus)
    return overlap / math.sqrt(len(q) * max(len(corpus), 1))


@dataclass
class _IndexedRow:
    memory: Memory
    embedding: list[float]


@dataclass
class _CacheEntry:
    value: bytes
    expires_at: float | None


@dataclass
class SharedMemoryState:
    """Process-local shard shared by memory + vector adapters (single-writer assumed)."""

    rows: dict[UUID, _IndexedRow] = field(default_factory=dict)
    by_user: dict[str, list[UUID]] = field(default_factory=dict)

    def _remember_user(self, user_id: str, mid: UUID) -> None:
        bucket = self.by_user.setdefault(user_id, [])
        if mid not in bucket:
            bucket.append(mid)

    def _forget_user(self, user_id: str, mid: UUID) -> None:
        bucket = self.by_user.get(user_id)
        if not bucket:
            return
        self.by_user[user_id] = [x for x in bucket if x != mid]


class InMemoryMemoryStore(MemoryStore):
    """Structured memory rows backed by :class:`SharedMemoryState`."""

    def __init__(self, state: SharedMemoryState) -> None:
        self._s = state

    async def upsert(self, memory: Memory) -> None:
        mid = memory.id
        existing = self._s.rows.get(mid)
        emb = existing.embedding if existing else (memory.embedding or [])
        row = _IndexedRow(memory=memory.model_copy(deep=True), embedding=emb)
        self._s.rows[mid] = row
        self._s._remember_user(memory.user_id, mid)

    async def get(self, memory_id: UUID) -> Memory | None:
        row = self._s.rows.get(memory_id)
        return row.memory.model_copy(deep=True) if row else None

    async def delete(self, memory_id: UUID) -> None:
        row = self._s.rows.pop(memory_id, None)
        if row:
            self._s._forget_user(row.memory.user_id, memory_id)

    async def list_for_user(
        self,
        user_id: str,
        *,
        limit: int = 10_000,
        include_archived: bool = False,
    ) -> list[Memory]:
        ids = list(self._s.by_user.get(user_id, []))
        out: list[Memory] = []
        for mid in ids:
            row = self._s.rows.get(mid)
            if not row:
                continue
            if not include_archived and row.memory.archived:
                continue
            out.append(row.memory.model_copy(deep=True))
            if len(out) >= limit:
                break
        return out

    async def keyword_search(self, user_id: str, query: str, *, limit: int) -> list[Memory]:
        scored: list[tuple[float, Memory]] = []
        for mid in self._s.by_user.get(user_id, []):
            row = self._s.rows.get(mid)
            if not row or row.memory.archived:
                continue
            s = _keyword_score(query, row.memory)
            if s > 0:
                scored.append((s, row.memory.model_copy(deep=True)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[:limit]]

    async def hierarchy_scope(self, user_id: str, path_prefix: str, *, limit: int) -> list[Memory]:
        prefix = normalize_hierarchy_path(path_prefix)
        out: list[Memory] = []
        for mid in self._s.by_user.get(user_id, []):
            row = self._s.rows.get(mid)
            if not row or row.memory.archived:
                continue
            hp = row.memory.hierarchy_path
            if hp == prefix or hp.startswith(prefix.rstrip("/") + "/") or hp.startswith(prefix):
                out.append(row.memory.model_copy(deep=True))
            if len(out) >= limit:
                break
        return out


class InMemoryVectorStore(VectorStore):
    """Cosine similarity search over rows in :class:`SharedMemoryState`."""

    def __init__(self, state: SharedMemoryState) -> None:
        self._s = state

    async def upsert(self, memory: Memory, embedding: list[float]) -> None:
        mid = memory.id
        row = self._s.rows.get(mid)
        mem = memory.model_copy(deep=True)
        mem.embedding = embedding
        self._s.rows[mid] = _IndexedRow(memory=mem, embedding=embedding)
        self._s._remember_user(mem.user_id, mid)

    async def delete(self, memory_id: UUID) -> None:
        self._s.rows.pop(memory_id, None)

    async def search(
        self,
        query_embedding: list[float],
        *,
        user_id: str,
        top_k: int,
        hierarchy_prefix: str | None = None,
        memory_types: frozenset[MemoryType] | None = None,
        session_id: str | None = None,
    ) -> list[Memory]:
        prefix = normalize_hierarchy_path(hierarchy_prefix) if hierarchy_prefix else None
        scored: list[tuple[float, Memory]] = []
        for mid in self._s.by_user.get(user_id, []):
            row = self._s.rows.get(mid)
            if not row or row.memory.archived:
                continue
            if prefix and not (
                row.memory.hierarchy_path == prefix
                or row.memory.hierarchy_path.startswith(prefix.rstrip("/") + "/")
                or row.memory.hierarchy_path.startswith(prefix)
            ):
                continue
            if memory_types is not None and row.memory.memory_type not in memory_types:
                continue
            if session_id is not None and row.memory.session_id != session_id:
                continue
            sim = _cosine_sim(query_embedding, row.embedding)
            scored.append((sim, row.memory.model_copy(deep=True)))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[: max(0, top_k)]]


class InMemoryGraphStore(GraphStore):
    """Adjacency list for memory edges (in-process)."""

    def __init__(self) -> None:
        self._edges: dict[UUID, MemoryEdge] = {}
        self._adj: dict[UUID, list[UUID]] = {}

    def _link(self, a: UUID, b: UUID) -> None:
        lst = self._adj.setdefault(a, [])
        if b not in lst:
            lst.append(b)

    async def add_edge(self, edge: MemoryEdge) -> None:
        self._edges[edge.id] = edge
        self._link(edge.source_id, edge.target_id)
        self._link(edge.target_id, edge.source_id)

    async def remove_edge(self, edge_id: UUID) -> None:
        edge = self._edges.pop(edge_id, None)
        if not edge:
            return
        for key, other in ((edge.source_id, edge.target_id), (edge.target_id, edge.source_id)):
            lst = self._adj.get(key)
            if not lst:
                continue
            self._adj[key] = [x for x in lst if x != other]

    async def neighbors(self, memory_id: UUID, *, direction: str = "both", limit: int = 64) -> list[UUID]:
        _ = direction  # undirected expansion for palace prototype
        out: list[UUID] = []
        seen: set[UUID] = set()
        for nid in self._adj.get(memory_id, []):
            if nid in seen:
                continue
            seen.add(nid)
            out.append(nid)
            if len(out) >= limit:
                break
        return out


class InMemoryCacheStore(CacheStore):
    """TTL-aware bytes cache (embeddings, working-memory buffers)."""

    def __init__(self) -> None:
        self._data: dict[str, _CacheEntry] = {}

    async def get(self, key: str) -> bytes | None:
        ent = self._data.get(key)
        if ent is None:
            return None
        if ent.expires_at is not None and time.monotonic() > ent.expires_at:
            self._data.pop(key, None)
            return None
        return ent.value

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:
        exp = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        self._data[key] = _CacheEntry(value=value, expires_at=exp)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

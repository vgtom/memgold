"""In-memory implementations for development and tests."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from memgold.embeddings.stub import stub_embed
from memgold.storage.base import KVStore, VectorStore

if TYPE_CHECKING:
    from memgold.models.memory import Memory


@dataclass
class _VectorRow:
    """Internal row for in-memory vector storage."""

    memory: "Memory"
    embedding: list[float]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    """Naive in-memory vector index keyed by user and cosine similarity."""

    def __init__(self) -> None:
        self._rows: dict[str, list[_VectorRow]] = {}

    async def add(self, memory: "Memory") -> None:
        embedding = stub_embed(memory.content)
        uid = memory.user_id
        row = _VectorRow(memory=memory, embedding=embedding)
        self._rows.setdefault(uid, []).append(row)

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        user_id: str,
    ) -> list["Memory"]:
        rows = self._rows.get(user_id, [])
        scored: list[tuple[float, "Memory"]] = []
        for row in rows:
            sim = _cosine_sim(query_embedding, row.embedding)
            scored.append((sim, row.memory))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored[: max(0, top_k)]]


class InMemoryKVStore(KVStore):
    """Simple dict-backed KV store with string values."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def set(self, key: str, value: str) -> None:
        self._data[key] = value

    async def get(self, key: str) -> str | None:
        return self._data.get(key)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)


def memory_to_json(memory: "Memory") -> str:
    """Serialize *memory* to JSON for KV storage."""
    return json.dumps(memory.model_dump(mode="json"))


def memory_from_json(data: str) -> "Memory":
    """Deserialize *Memory* from JSON string."""
    from memgold.models.memory import Memory

    return Memory.model_validate_json(data)

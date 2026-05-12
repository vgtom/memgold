"""In-memory adapters and JSON helpers (compatibility exports)."""

from __future__ import annotations

from memgold.storage.in_memory import (
    InMemoryCacheStore,
    InMemoryGraphStore,
    InMemoryMemoryStore,
    InMemoryVectorStore,
    SharedMemoryState,
)
from memgold.storage.serialization import memory_from_json, memory_to_json

__all__ = [
    "InMemoryCacheStore",
    "InMemoryGraphStore",
    "InMemoryMemoryStore",
    "InMemoryVectorStore",
    "SharedMemoryState",
    "memory_from_json",
    "memory_to_json",
]

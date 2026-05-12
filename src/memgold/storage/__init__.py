"""Storage backends and in-memory adapters."""

from memgold.interfaces.storage import CacheStore, GraphStore, MemoryStore, VectorStore
from memgold.storage.in_memory import (
    InMemoryCacheStore,
    InMemoryGraphStore,
    InMemoryMemoryStore,
    InMemoryVectorStore,
    SharedMemoryState,
)
from memgold.storage.serialization import memory_from_json, memory_to_json

__all__ = [
    "CacheStore",
    "GraphStore",
    "InMemoryCacheStore",
    "InMemoryGraphStore",
    "InMemoryMemoryStore",
    "InMemoryVectorStore",
    "MemoryStore",
    "SharedMemoryState",
    "VectorStore",
    "memory_from_json",
    "memory_to_json",
]

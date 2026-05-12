"""Storage port re-exports (back-compat shim for older imports)."""

from __future__ import annotations

from memgold.interfaces.storage import CacheStore, GraphStore, MemoryStore, VectorStore

__all__ = ["CacheStore", "GraphStore", "MemoryStore", "VectorStore"]

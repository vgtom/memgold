"""Vector-backed retrieval with stub embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from memgold.embeddings.stub import stub_embed

if TYPE_CHECKING:
    from memgold.models.memory import Memory
    from memgold.storage.base import VectorStore


class MemoryRetriever:
    """Retrieve memories by semantic similarity using the configured vector index."""

    def __init__(self, vector_store: "VectorStore", default_top_k: int = 10) -> None:
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    async def retrieve(
        self,
        query: str,
        user_id: str,
        *,
        top_k: int | None = None,
    ) -> list["Memory"]:
        """Embed *query* and query the vector store for *user_id*."""
        k = top_k if top_k is not None else self._default_top_k
        query_embedding = stub_embed(query)
        return await self._vector_store.search(query_embedding, k, user_id)

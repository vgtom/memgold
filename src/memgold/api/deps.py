"""FastAPI dependency wiring for application services."""

from __future__ import annotations

from dataclasses import dataclass

from memgold.extraction.extractor import MemoryExtractor
from memgold.ranking.ranker import MemoryRanker
from memgold.retrieval.retriever import MemoryRetriever
from memgold.services.context import ContextBuilder
from memgold.services.read import MemoryReadService
from memgold.services.write import MemoryWriteService
from memgold.storage.memory_stores import InMemoryKVStore, InMemoryVectorStore


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Composition root for storage backends and domain services."""

    vector_store: InMemoryVectorStore
    kv_store: InMemoryKVStore
    extractor: MemoryExtractor
    retriever: MemoryRetriever
    ranker: MemoryRanker
    write_service: MemoryWriteService
    read_service: MemoryReadService
    context_builder: ContextBuilder


def build_container() -> AppContainer:
    """Construct the default in-memory stack used by the API process."""
    vector_store = InMemoryVectorStore()
    kv_store = InMemoryKVStore()
    extractor = MemoryExtractor()
    retriever = MemoryRetriever(vector_store)
    ranker = MemoryRanker()
    write_service = MemoryWriteService(extractor, vector_store, kv_store)
    read_service = MemoryReadService(retriever, ranker)
    context_builder = ContextBuilder(read_service)
    return AppContainer(
        vector_store=vector_store,
        kv_store=kv_store,
        extractor=extractor,
        retriever=retriever,
        ranker=ranker,
        write_service=write_service,
        read_service=read_service,
        context_builder=context_builder,
    )


_CONTAINER: AppContainer | None = None


def get_container() -> AppContainer:
    """Return the process-wide container (lazy singleton)."""
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = build_container()
    return _CONTAINER


def get_write_service() -> MemoryWriteService:
    return get_container().write_service


def get_read_service() -> MemoryReadService:
    return get_container().read_service


def get_context_builder() -> ContextBuilder:
    return get_container().context_builder

"""FastAPI dependency wiring for application services."""

from __future__ import annotations

from dataclasses import dataclass

from memgold.config.settings import MemoryLayerSettings, load_settings
from memgold.embeddings.cached import CachedEmbedder
from memgold.embeddings.stub import StubEmbedder
from memgold.extraction.heuristic import DefaultExtractionPipeline
from memgold.observability.logging import configure_logging
from memgold.ranking.pipeline import RankingPipeline
from memgold.retrieval.hybrid import HybridRetriever
from memgold.services.consolidation import ConsolidationService
from memgold.services.context import ContextBuilder
from memgold.services.decay import DecayService
from memgold.services.embedding_service import EmbeddingService
from memgold.services.ingestion import MemoryIngestionService
from memgold.services.ranking_service import RankingService
from memgold.services.read import MemoryReadService
from memgold.services.reflection import ReflectionService
from memgold.services.retrieval_service import RetrievalService
from memgold.services.working_memory import WorkingMemoryBuffer
from memgold.services.write import MemoryWriteService
from memgold.storage.in_memory import (
    InMemoryCacheStore,
    InMemoryGraphStore,
    InMemoryMemoryStore,
    InMemoryVectorStore,
    SharedMemoryState,
)


@dataclass(frozen=True, slots=True)
class AppContainer:
    """Composition root for storage backends and domain services."""

    settings: MemoryLayerSettings
    state: SharedMemoryState
    memory_store: InMemoryMemoryStore
    vector_store: InMemoryVectorStore
    graph_store: InMemoryGraphStore
    cache_store: InMemoryCacheStore
    embedding_service: EmbeddingService
    pipeline: DefaultExtractionPipeline
    ingestion: MemoryIngestionService
    hybrid_retriever: HybridRetriever
    ranking_pipeline: RankingPipeline
    ranking_service: RankingService
    retrieval_service: RetrievalService
    write_service: MemoryWriteService
    read_service: MemoryReadService
    context_builder: ContextBuilder
    consolidation_service: ConsolidationService
    reflection_service: ReflectionService
    decay_service: DecayService


def build_container() -> AppContainer:
    """Construct the default in-memory stack used by the API process."""
    settings = load_settings()
    configure_logging(settings)

    state = SharedMemoryState()
    memory_store = InMemoryMemoryStore(state)
    vector_store = InMemoryVectorStore(state)
    graph_store = InMemoryGraphStore()
    cache_store = InMemoryCacheStore()

    embedder = CachedEmbedder(StubEmbedder(), cache_store)
    embedding_service = EmbeddingService(embedder, settings)

    pipeline = DefaultExtractionPipeline()
    working_memory = WorkingMemoryBuffer(cache_store, settings)
    ingestion = MemoryIngestionService(
        pipeline=pipeline,
        memory_store=memory_store,
        vector_store=vector_store,
        graph_store=graph_store,
        embedding_service=embedding_service,
        settings=settings,
        working_memory=working_memory,
    )

    hybrid = HybridRetriever(
        vector_store=vector_store,
        memory_store=memory_store,
        graph_store=graph_store,
        embedding_service=embedding_service,
        settings=settings,
    )
    ranking_pipeline = RankingPipeline()
    ranking_service = RankingService(ranking_pipeline)
    retrieval = RetrievalService(
        retriever=hybrid,
        ranking=ranking_service,
        memory_store=memory_store,
    )

    write_service = MemoryWriteService(ingestion)
    read_service = MemoryReadService(retrieval)
    context_builder = ContextBuilder(read_service)

    consolidation = ConsolidationService(
        memory_store=memory_store,
        vector_store=vector_store,
        embedding_service=embedding_service,
        settings=settings,
    )
    reflection = ReflectionService(
        memory_store=memory_store,
        vector_store=vector_store,
        embedding_service=embedding_service,
        settings=settings,
    )
    decay = DecayService(memory_store, settings)

    return AppContainer(
        settings=settings,
        state=state,
        memory_store=memory_store,
        vector_store=vector_store,
        graph_store=graph_store,
        cache_store=cache_store,
        embedding_service=embedding_service,
        pipeline=pipeline,
        ingestion=ingestion,
        hybrid_retriever=hybrid,
        ranking_pipeline=ranking_pipeline,
        ranking_service=ranking_service,
        retrieval_service=retrieval,
        write_service=write_service,
        read_service=read_service,
        context_builder=context_builder,
        consolidation_service=consolidation,
        reflection_service=reflection,
        decay_service=decay,
    )


_CONTAINER: AppContainer | None = None


def get_container() -> AppContainer:
    """Return the process-wide container (lazy singleton)."""
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = build_container()
    return _CONTAINER


def reset_container() -> None:
    """Reset the DI container (primarily for tests)."""
    global _CONTAINER
    _CONTAINER = None


def get_write_service() -> MemoryWriteService:
    return get_container().write_service


def get_read_service() -> MemoryReadService:
    return get_container().read_service


def get_context_builder() -> ContextBuilder:
    return get_container().context_builder


def get_ingestion_service() -> MemoryIngestionService:
    return get_container().ingestion


def get_retrieval_service() -> RetrievalService:
    return get_container().retrieval_service


def get_consolidation_service() -> ConsolidationService:
    return get_container().consolidation_service


def get_reflection_service() -> ReflectionService:
    return get_container().reflection_service


def get_decay_service() -> DecayService:
    return get_container().decay_service

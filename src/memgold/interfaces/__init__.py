"""Abstract interfaces (ports) for dependency injection."""

from memgold.interfaces.embedder import Embedder
from memgold.interfaces.extraction import (
    EntityExtractor,
    ExtractionPipeline,
    FactExtractor,
    HierarchyClassifier,
    IntentExtractor,
    TopicExtractor,
)
from memgold.interfaces.storage import CacheStore, GraphStore, MemoryStore, VectorStore

__all__ = [
    "CacheStore",
    "Embedder",
    "EntityExtractor",
    "ExtractionPipeline",
    "FactExtractor",
    "GraphStore",
    "HierarchyClassifier",
    "IntentExtractor",
    "MemoryStore",
    "TopicExtractor",
    "VectorStore",
]

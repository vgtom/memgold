"""Extraction pipeline ports (pluggable stages)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from memgold.models.extraction import (
    ExtractedEntity,
    ExtractedFact,
    HierarchyAssignment,
    MemoryCandidate,
    TopicHypothesis,
)


class FactExtractor(ABC):
    """Extract atomic facts from unstructured text."""

    @abstractmethod
    async def extract_facts(self, text: str, *, user_id: str) -> list[ExtractedFact]:
        """Return candidate facts."""


class EntityExtractor(ABC):
    """Extract named entities."""

    @abstractmethod
    async def extract_entities(self, text: str, *, user_id: str) -> list[ExtractedEntity]:
        """Return entities with coarse labels."""


class TopicExtractor(ABC):
    """Surface topical labels for clustering and routing."""

    @abstractmethod
    async def extract_topics(self, text: str, *, user_id: str) -> list[TopicHypothesis]:
        """Return scored topic hypotheses."""


class IntentExtractor(ABC):
    """Optional intent / task labels."""

    @abstractmethod
    async def extract_intents(self, text: str, *, user_id: str) -> list[TopicHypothesis]:
        """Return scored intent hypotheses (reuses ``TopicHypothesis`` shape)."""


class HierarchyClassifier(ABC):
    """Assign dynamic palace paths from topics/entities."""

    @abstractmethod
    async def classify_path(
        self,
        text: str,
        *,
        user_id: str,
        topics: list[TopicHypothesis],
        entities: list[ExtractedEntity],
    ) -> HierarchyAssignment:
        """Return normalized hierarchy path suggestion."""


class ExtractionPipeline(ABC):
    """Orchestrates extractors into ``MemoryCandidate`` records."""

    @abstractmethod
    async def build_candidates(
        self,
        text: str,
        *,
        user_id: str,
        session_id: str | None,
    ) -> list[MemoryCandidate]:
        """Produce memory candidates ready for persistence policies."""

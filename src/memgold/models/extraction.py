"""Structured outputs from pluggable extractors."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from memgold.models.enums import GraphRelationType, MemorySource, MemoryType


class ExtractedFact(BaseModel):
    """Atomic factual statement extracted from text."""

    text: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class ExtractedEntity(BaseModel):
    """Named entity mention."""

    name: str
    label: str = "ENTITY"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TopicHypothesis(BaseModel):
    """Topic / intent labels used for palace routing."""

    label: str
    score: float = Field(ge=0.0, le=1.0)


class HierarchyAssignment(BaseModel):
    """Classifier output for dynamic palace placement."""

    path: str = "/misc/general"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryCandidate(BaseModel):
    """Pre-persist memory suggestion assembled by the extraction pipeline."""

    content: str
    summary: str | None = None
    memory_type: MemoryType
    hierarchy_path: str
    semantic_cluster: str | None = None
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    salience_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source: MemorySource = MemorySource.USER_MESSAGE
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContradictionSignal(BaseModel):
    """Heuristic flag when a new candidate may conflict with an existing memory."""

    existing_memory_id: UUID
    new_text: str
    existing_text: str
    score: float = Field(ge=0.0, le=1.0, description="Higher means stronger suspected conflict.")


class LinkSuggestion(BaseModel):
    """Suggested graph edge produced during ingestion or consolidation."""

    source_id: UUID
    target_id: UUID
    relation: GraphRelationType
    weight: float = Field(default=0.8, ge=0.0, le=1.0)

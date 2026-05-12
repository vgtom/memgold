"""HTTP request and response models (transport layer)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from memgold.models.enums import MemoryType
from memgold.models.memory import Memory


class MemoryWriteRequest(BaseModel):
    """Payload for ingesting new conversational text."""

    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class MemoryReadRequest(BaseModel):
    """Payload for querying stored memories."""

    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class MemoryContextRequest(BaseModel):
    """Payload for building ranked context for prompting."""

    user_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class MemoryListResponse(BaseModel):
    """Standard envelope for ranked or extracted memories."""

    memories: list[Memory]


class MemoryContextResponse(BaseModel):
    """Structured context bundle."""

    memories: list[Memory]
    context_string: str


class CreateMemoryRequest(BaseModel):
    """Create memories from raw conversational text."""

    user_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    session_id: str | None = None


class MemorySearchParams(BaseModel):
    """Query parameters for hybrid search."""

    user_id: str = Field(..., min_length=1)
    q: str = Field(..., min_length=1)
    top_k: int = Field(default=12, ge=1, le=100)
    hierarchy_prefix: str | None = None
    session_id: str | None = None
    expand_graph: bool = True


class ConsolidateMemoriesRequest(BaseModel):
    """Trigger palace-level consolidation for a user."""

    user_id: str = Field(..., min_length=1)


class SummarizeMemoriesRequest(BaseModel):
    """Summarize an explicit subset of memory ids."""

    user_id: str = Field(..., min_length=1)
    memory_ids: list[UUID] = Field(default_factory=list)


class ReflectMemoriesRequest(BaseModel):
    """Trigger a reflection job for a user."""

    user_id: str = Field(..., min_length=1)
    lookback: int = Field(default=24, ge=1, le=500)


class SingleMemoryResponse(BaseModel):
    """Optional envelope for single-memory endpoints."""

    memory: Memory | None = None


class JobMemoriesResponse(BaseModel):
    """Envelope for batch job outputs."""

    memories: list[Memory]

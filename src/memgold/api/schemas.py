"""HTTP request and response models (transport layer)."""

from __future__ import annotations

from pydantic import BaseModel, Field

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

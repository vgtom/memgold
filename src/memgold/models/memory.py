"""Core memory schema."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """High-level classification for a stored memory."""

    FACT = "fact"
    PREFERENCE = "preference"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Memory(BaseModel):
    """A single unit of persisted memory.

    Attributes:
        id: Stable identifier for this memory row.
        user_id: Tenant / user scope for isolation.
        content: Raw natural-language memory text.
        type: Semantic category (fact, preference, episodic, semantic).
        confidence_score: Extractor or fusion confidence in ``[0, 1]``.
        tags: Optional structured labels for filtering or routing.
        created_at: Creation timestamp (UTC).
        last_accessed_at: Last retrieval touch time (UTC); optional until read path updates it.
        metadata: Arbitrary extension payload (source ids, provenance, etc.).
    """

    id: UUID = Field(default_factory=uuid4)
    user_id: str = ""
    content: str = ""
    type: MemoryType = MemoryType.SEMANTIC
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utc_now)
    last_accessed_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

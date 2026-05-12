"""Canonical memory record for the AI memory layer."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from memgold.models.enums import MemorySource, MemoryType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_hierarchy_path(path: str) -> str:
    """Return a stable palace path: lowercase segments, leading ``/``, no trailing ``/``."""
    raw = (path or "").strip().lower()
    if not raw:
        return "/misc/general"
    parts = [p for p in raw.replace("\\", "/").split("/") if p]
    if not parts:
        return "/misc/general"
    return "/" + "/".join(parts)


class Memory(BaseModel):
    """Single persisted memory unit (palace-organized, typed, retrievable).

    Embeddings are optional on the in-memory instance; long-term persistence
    should store vectors in a ``VectorStore`` and metadata in a ``MemoryStore``.
    """

    id: UUID = Field(default_factory=uuid4)
    user_id: str = ""
    session_id: str | None = None

    content: str = ""
    summary: str | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC

    hierarchy_path: str = Field(default="/misc/general")
    semantic_cluster: str | None = None

    embedding: list[float] | None = Field(
        default=None,
        description="Dense vector; excluded from default JSON persistence.",
    )

    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    salience_score: float = Field(default=0.5, ge=0.0, le=1.0)
    decay_score: float = Field(default=1.0, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)

    linked_memory_ids: list[UUID] = Field(default_factory=list)

    source: MemorySource = MemorySource.UNKNOWN
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    archived: bool = Field(default=False, description="Soft-archive for pruning workflows.")

    @field_validator("hierarchy_path", mode="before")
    @classmethod
    def _coerce_path(cls, v: Any) -> str:
        if v is None:
            return normalize_hierarchy_path("")
        if isinstance(v, str):
            return normalize_hierarchy_path(v)
        return normalize_hierarchy_path(str(v))

    def touch_access(self) -> None:
        """Increment access counter and bump ``updated_at`` (mutates in place)."""
        self.access_count += 1
        self.updated_at = _utc_now()

    model_config = {"frozen": False}

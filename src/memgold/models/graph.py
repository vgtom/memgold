"""Graph edge models for memory linking."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from memgold.models.enums import GraphRelationType


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryEdge(BaseModel):
    """Directed relationship between two domain nodes.

    Nodes may be memories or synthetic event ids (string targets).
    """

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    target_id: UUID
    relation: GraphRelationType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    model_config = {"frozen": False}

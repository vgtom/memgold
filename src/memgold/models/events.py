"""Ingestion events for episodic provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IngestionEvent(BaseModel):
    """Raw conversational or system event prior to extraction."""

    id: UUID = Field(default_factory=uuid4)
    user_id: str
    session_id: str | None = None
    role: str = "user"
    text: str
    occurred_at: datetime = Field(default_factory=_utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}

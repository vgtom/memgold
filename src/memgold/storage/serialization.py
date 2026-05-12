"""Persistence helpers for memory records."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memgold.models.memory import Memory


def memory_to_json(memory: "Memory", *, include_embedding: bool = False) -> str:
    """Serialize *memory* to JSON for KV/backup adapters.

    By default embeddings are stripped to keep payloads small.
    """
    mode: dict[str, Any] = {"mode": "json"}
    if not include_embedding:
        mode["exclude"] = {"embedding"}
    return memory.model_dump_json(**mode)


def memory_from_json(data: str) -> "Memory":
    """Deserialize :class:`~memgold.models.memory.Memory` from JSON."""
    from memgold.models.memory import Memory

    return Memory.model_validate_json(data)

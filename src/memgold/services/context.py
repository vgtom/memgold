"""Assemble LLM-ready context strings from ranked memories."""

from __future__ import annotations

from typing import Any

from memgold.models.memory import Memory
from memgold.services.read import MemoryReadService


class ContextBuilder:
    """Formats retrieved memories into prompt-friendly bundles."""

    def __init__(self, read_service: MemoryReadService) -> None:
        self._read_service = read_service

    async def build_context(self, user_id: str, query: str) -> dict[str, Any]:
        """Return structured memories plus a compact newline context string."""
        memories: list[Memory] = await self._read_service.retrieve(user_id, query)
        lines = [m.content for m in memories if m.content.strip()]
        context_string = "\n".join(f"- {line}" for line in lines)
        return {"memories": memories, "context_string": context_string}

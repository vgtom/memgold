"""Token-aware sliding window for short-term / working memory."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.storage import CacheStore


def _estimate_tokens(text: str) -> int:
    """Rough token estimate without tiktoken (fast path)."""
    return max(1, len(text) // 4)


class WorkingMemoryBuffer:
    """Stores recent conversational turns per session in a :class:`CacheStore`."""

    def __init__(self, cache: "CacheStore", settings: "MemoryLayerSettings") -> None:
        self._cache = cache
        self._settings = settings

    def _key(self, user_id: str, session_id: str) -> str:
        return f"wm:{user_id}:{session_id}"

    async def append_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a message and trim the window to policy limits."""
        key = self._key(user_id, session_id)
        raw = await self._cache.get(key)
        messages: list[dict[str, Any]] = []
        if raw:
            messages = json.loads(raw.decode("utf-8"))
        messages.append({"role": role, "text": text, "meta": metadata or {}})
        # trim by message count
        max_m = self._settings.working_memory_max_messages
        if len(messages) > max_m:
            messages = messages[-max_m:]
        # trim by rough tokens from the left
        max_t = self._settings.working_memory_max_tokens_estimate
        total = sum(_estimate_tokens(m["text"]) for m in messages)
        while messages and total > max_t:
            removed = messages.pop(0)
            total -= _estimate_tokens(removed["text"])
        await self._cache.set(key, json.dumps(messages).encode("utf-8"), ttl_seconds=86400)

    async def window(self, *, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """Return the buffered messages (oldest first)."""
        raw = await self._cache.get(self._key(user_id, session_id))
        if not raw:
            return []
        return json.loads(raw.decode("utf-8"))

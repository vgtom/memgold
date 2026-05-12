"""Decay, archival, and lifecycle maintenance."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from memgold.observability.logging import log_event

if TYPE_CHECKING:
    from memgold.config.settings import MemoryLayerSettings
    from memgold.interfaces.storage import MemoryStore


class DecayService:
    """Applies exponential decay on ``decay_score`` and archives low-value rows."""

    def __init__(self, memory_store: "MemoryStore", settings: "MemoryLayerSettings") -> None:
        self._memory_store = memory_store
        self._settings = settings

    async def apply_decay(self, *, user_id: str) -> int:
        """Age memories for *user_id* based on ``decay_half_life_days``.

        Returns the number of memories updated.
        """
        mems = await self._memory_store.list_for_user(user_id, limit=50_000, include_archived=False)
        now = datetime.now(timezone.utc)
        half_life_seconds = max(1.0, float(self._settings.decay_half_life_days) * 86400.0)
        updated = 0
        for m in mems:
            ref = m.updated_at or m.created_at
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (now - ref).total_seconds())
            factor = math.pow(0.5, age_seconds / half_life_seconds)
            # preserve high-salience traces longer
            sal = float(m.salience_score)
            adjusted = float(m.decay_score) * (0.65 * factor + 0.35 * sal)
            m.decay_score = max(0.0, min(1.0, adjusted))
            if m.decay_score < self._settings.archive_decay_threshold and sal < 0.35:
                m.archived = True
            await self._memory_store.upsert(m)
            updated += 1
        log_event("memory_decay", {"user_id": user_id, "updated": updated}, settings=self._settings)
        return updated

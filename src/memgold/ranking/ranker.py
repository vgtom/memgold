"""Re-rank retrieved memories using lightweight heuristics."""

from __future__ import annotations

from datetime import datetime, timezone

from memgold.models.memory import Memory


class MemoryRanker:
    """Combine confidence and recency into a single ranking score."""

    def __init__(
        self,
        *,
        confidence_weight: float = 0.6,
        recency_weight: float = 0.4,
        max_age_seconds: float = 365.0 * 24 * 3600,
    ) -> None:
        self._wc = confidence_weight
        self._wr = recency_weight
        self._max_age = max_age_seconds

    def _recency_score(self, reference: datetime, now: datetime) -> float:
        """Map age to ``[0, 1]`` where newer references score higher."""
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        delta = max(0.0, (now - reference).total_seconds())
        # Clamp: brand-new => 1.0, older than max_age => ~0.0
        return max(0.0, 1.0 - min(delta / self._max_age, 1.0))

    async def rank(self, memories: list["Memory"]) -> list["Memory"]:
        """Return *memories* sorted by descending combined score."""
        if not memories:
            return []
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, Memory]] = []
        for m in memories:
            ref = m.last_accessed_at or m.created_at
            if ref.tzinfo is None:
                ref = ref.replace(tzinfo=timezone.utc)
            rec = self._recency_score(ref, now)
            score = self._wc * float(m.confidence_score) + self._wr * rec
            scored.append((score, m))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

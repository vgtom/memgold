"""Re-ranking pipeline with personalization hooks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from memgold.models.enums import MemoryType
from memgold.models.memory import Memory
from memgold.retrieval.types import ScoredMemory

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(slots=True)
class RankingWeights:
    """Weights for second-stage ranking after hybrid fusion."""

    retrieval: float = 0.55
    recency: float = 0.15
    salience: float = 0.15
    confidence: float = 0.1
    type_bias: float = 0.05


class RankingPipeline:
    """Normalize sub-scores and apply memory-type personalization."""

    def __init__(
        self,
        weights: RankingWeights | None = None,
        *,
        type_multipliers: Mapping[MemoryType, float] | None = None,
        max_age_seconds: float = 365.0 * 24 * 3600,
    ) -> None:
        self._w = weights or RankingWeights()
        self._max_age = max_age_seconds
        self._type_mult: dict[MemoryType, float] = dict(type_multipliers or {})
        defaults: dict[MemoryType, float] = {
            MemoryType.SEMANTIC: 1.0,
            MemoryType.EPISODIC: 0.95,
            MemoryType.PROCEDURAL: 1.05,
            MemoryType.SHORT_TERM: 0.85,
        }
        for k, v in defaults.items():
            self._type_mult.setdefault(k, v)

    def _recency(self, mem: Memory) -> float:
        ref = mem.updated_at or mem.created_at
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = max(0.0, (now - ref).total_seconds())
        return max(0.0, 1.0 - min(age / self._max_age, 1.0))

    def rank(
        self,
        scored: list[tuple[Memory, ScoredMemory]],
        *,
        user_persona_boost: Mapping[UUID, float] | None = None,
    ) -> list[Memory]:
        """Return memories sorted by final rank score."""
        persona = user_persona_boost or {}
        ranked: list[tuple[float, Memory]] = []
        for mem, sc in scored:
            r = self._recency(mem)
            t_mult = self._type_mult.get(mem.memory_type, 1.0)
            p_boost = float(persona.get(mem.id, 1.0))
            score = (
                self._w.retrieval * float(sc.fused_score)
                + self._w.recency * r
                + self._w.salience * float(mem.salience_score)
                + self._w.confidence * float(mem.confidence)
                + self._w.type_bias * t_mult
            ) * p_boost
            ranked.append((score, mem))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in ranked]

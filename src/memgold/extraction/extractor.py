"""Back-compat extractor facade over :class:`~memgold.extraction.heuristic.DefaultExtractionPipeline`."""

from __future__ import annotations

from memgold.extraction.heuristic import DefaultExtractionPipeline
from memgold.interfaces.extraction import ExtractionPipeline
from memgold.models.memory import Memory


class MemoryExtractor:
    """Deprecated surface; prefer :class:`~memgold.services.ingestion.MemoryIngestionService`."""

    def __init__(self, pipeline: ExtractionPipeline | None = None) -> None:
        self._pipeline = pipeline or DefaultExtractionPipeline()

    async def extract(
        self,
        text: str,
        *,
        user_id: str = "anonymous",
        session_id: str | None = None,
    ) -> list[Memory]:
        """Return :class:`~memgold.models.memory.Memory` rows without persistence."""
        from memgold.extraction.importance import score_importance
        from memgold.models.extraction import MemoryCandidate
        from uuid import uuid4

        candidates = await self._pipeline.build_candidates(
            text,
            user_id=user_id,
            session_id=session_id,
        )
        out: list[Memory] = []
        for cand in candidates:
            assert isinstance(cand, MemoryCandidate)
            salience = score_importance(cand, text_len=len(text))
            out.append(
                Memory(
                    id=uuid4(),
                    user_id=user_id,
                    session_id=session_id,
                    content=cand.content,
                    summary=cand.summary,
                    memory_type=cand.memory_type,
                    hierarchy_path=cand.hierarchy_path,
                    semantic_cluster=cand.semantic_cluster,
                    entities=list(cand.entities),
                    keywords=list(cand.keywords),
                    confidence=float(cand.confidence),
                    salience_score=float(max(cand.salience_score, salience)),
                    source=cand.source,
                    metadata=dict(cand.metadata),
                ),
            )
        return out

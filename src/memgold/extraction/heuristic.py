"""Heuristic extractors and default pipeline (no LLM required)."""

from __future__ import annotations

import re
from collections import Counter

from memgold.interfaces.extraction import (
    EntityExtractor,
    ExtractionPipeline,
    FactExtractor,
    HierarchyClassifier,
    IntentExtractor,
    TopicExtractor,
)
from memgold.models.enums import MemorySource, MemoryType
from memgold.models.extraction import (
    ExtractedEntity,
    ExtractedFact,
    HierarchyAssignment,
    MemoryCandidate,
    TopicHypothesis,
)
from memgold.models.memory import normalize_hierarchy_path

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class HeuristicFactExtractor(FactExtractor):
    """Split sentences into lightweight fact candidates."""

    async def extract_facts(self, text: str, *, user_id: str) -> list[ExtractedFact]:
        _ = user_id
        raw = (text or "").strip()
        if not raw:
            return []
        parts = [p.strip() for p in _SENT_SPLIT.split(raw) if p.strip()]
        if not parts:
            parts = [raw]
        facts: list[ExtractedFact] = []
        for p in parts[:8]:
            toks = [t.lower() for t in re.findall(r"[a-zA-Z]{3,}", p)]
            kw = list(dict.fromkeys(toks))[:12]
            facts.append(
                ExtractedFact(
                    text=p,
                    confidence=min(1.0, 0.55 + 0.05 * min(len(kw), 5)),
                    entities=[],
                    keywords=kw,
                ),
            )
        return facts


class HeuristicEntityExtractor(EntityExtractor):
    """Capitalized token runs as coarse entities."""

    async def extract_entities(self, text: str, *, user_id: str) -> list[ExtractedEntity]:
        _ = user_id
        hits = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text or "")
        uniq: list[str] = []
        for h in hits:
            if h not in uniq:
                uniq.append(h)
        return [ExtractedEntity(name=n, label="MISC", confidence=0.5) for n in uniq[:16]]


class HeuristicTopicExtractor(TopicExtractor):
    """Bag-of-words pseudo topics."""

    async def extract_topics(self, text: str, *, user_id: str) -> list[TopicHypothesis]:
        _ = user_id
        toks = [t for t in re.findall(r"[a-zA-Z]{4,}", text or "")]
        counts = Counter(t.lower() for t in toks)
        top = counts.most_common(6)
        if not top:
            return [TopicHypothesis(label="general", score=0.4)]
        max_c = top[0][1]
        return [TopicHypothesis(label=w, score=min(1.0, c / max(max_c, 1))) for w, c in top]


class HeuristicIntentExtractor(IntentExtractor):
    """Map modals/imperatives to coarse intents."""

    async def extract_intents(self, text: str, *, user_id: str) -> list[TopicHypothesis]:
        _ = user_id
        low = (text or "").lower()
        intents: list[TopicHypothesis] = []
        if any(k in low for k in ("how do i", "how to", "steps", "guide")):
            intents.append(TopicHypothesis(label="how_to", score=0.62))
        if any(k in low for k in ("remember", "don't forget", "note that")):
            intents.append(TopicHypothesis(label="memorization", score=0.7))
        if any(k in low for k in ("prefer", "i like", "i hate")):
            intents.append(TopicHypothesis(label="preference", score=0.66))
        if not intents:
            intents.append(TopicHypothesis(label="statement", score=0.45))
        return intents


class HeuristicHierarchyClassifier(HierarchyClassifier):
    """Build palace path segments from topics + entities."""

    async def classify_path(
        self,
        text: str,
        *,
        user_id: str,
        topics: list[TopicHypothesis],
        entities: list[ExtractedEntity],
    ) -> HierarchyAssignment:
        _ = user_id, text
        parts: list[str] = []
        if entities:
            parts.append(entities[0].name.lower().replace(" ", "_")[:24])
        for th in sorted(topics, key=lambda t: t.score, reverse=True)[:3]:
            if th.label not in parts:
                parts.append(th.label[:24])
        if not parts:
            parts = ["general"]
        path = normalize_hierarchy_path("/life/" + "/".join(parts))
        return HierarchyAssignment(path=path, confidence=0.55)


class DefaultExtractionPipeline(ExtractionPipeline):
    """Compose heuristic extractors into :class:`MemoryCandidate` rows."""

    def __init__(
        self,
        *,
        facts: FactExtractor | None = None,
        entities: EntityExtractor | None = None,
        topics: TopicExtractor | None = None,
        intents: IntentExtractor | None = None,
        hierarchy: HierarchyClassifier | None = None,
    ) -> None:
        self._facts = facts or HeuristicFactExtractor()
        self._entities = entities or HeuristicEntityExtractor()
        self._topics = topics or HeuristicTopicExtractor()
        self._intents = intents or HeuristicIntentExtractor()
        self._hierarchy = hierarchy or HeuristicHierarchyClassifier()

    async def build_candidates(
        self,
        text: str,
        *,
        user_id: str,
        session_id: str | None,
    ) -> list[MemoryCandidate]:
        facts = await self._facts.extract_facts(text, user_id=user_id)
        entities = await self._entities.extract_entities(text, user_id=user_id)
        topics = await self._topics.extract_topics(text, user_id=user_id)
        intents = await self._intents.extract_intents(text, user_id=user_id)
        hier = await self._hierarchy.classify_path(
            text,
            user_id=user_id,
            topics=topics + intents,
            entities=entities,
        )
        cluster = topics[0].label if topics else None
        candidates: list[MemoryCandidate] = []

        # Episodic envelope for the raw turn
        candidates.append(
            MemoryCandidate(
                content=text.strip()[:4000],
                summary=f"Interaction snippet ({len(text)} chars)",
                memory_type=MemoryType.EPISODIC,
                hierarchy_path=hier.path,
                semantic_cluster=cluster,
                entities=[e.name for e in entities],
                keywords=[t.label for t in topics[:8]],
                confidence=0.5,
                salience_score=0.45,
                source=MemorySource.USER_MESSAGE,
                metadata={"session_id": session_id} if session_id else {},
            ),
        )

        for fact in facts[:6]:
            kw = list(dict.fromkeys(fact.keywords + [t.label for t in topics[:4]]))
            candidates.append(
                MemoryCandidate(
                    content=fact.text,
                    summary=None,
                    memory_type=MemoryType.SEMANTIC,
                    hierarchy_path=hier.path,
                    semantic_cluster=cluster,
                    entities=[e.name for e in entities][:8],
                    keywords=kw,
                    confidence=min(1.0, fact.confidence),
                    salience_score=min(1.0, 0.5 + 0.05 * len(kw)),
                    source=MemorySource.USER_MESSAGE,
                    metadata={"extractor": "HeuristicFactExtractor"},
                ),
            )

        if any("how_to" == i.label for i in intents):
            candidates.append(
                MemoryCandidate(
                    content=f"Procedure-oriented message: {text.strip()[:512]}",
                    summary="Potential procedural pattern",
                    memory_type=MemoryType.PROCEDURAL,
                    hierarchy_path=hier.path,
                    semantic_cluster="procedure",
                    entities=[e.name for e in entities],
                    keywords=[t.label for t in topics[:6]],
                    confidence=0.48,
                    salience_score=0.52,
                    source=MemorySource.USER_MESSAGE,
                    metadata={"extractor": "HeuristicIntentExtractor"},
                ),
            )

        return candidates

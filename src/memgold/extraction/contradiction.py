"""Lightweight contradiction / tension detection between memories."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from uuid import UUID

from memgold.models.enums import GraphRelationType
from memgold.models.extraction import ContradictionSignal, LinkSuggestion
from memgold.models.graph import MemoryEdge
from memgold.models.memory import Memory


_NEG_WORDS = (
    " not ",
    " no ",
    " never ",
    " don't ",
    " do not ",
    " isn't ",
    " aren't ",
    " wasn't ",
    " weren't ",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def detect_contradictions(candidate_text: str, existing: list[Memory]) -> list[ContradictionSignal]:
    """Return suspected contradictions using lexical negation + similarity.

    TODO: NLI model (entailment/contradiction) for robust fusion.
    """
    out: list[ContradictionSignal] = []
    c_norm = _normalize(candidate_text)
    if not c_norm:
        return out
    for mem in existing:
        m_norm = _normalize(mem.content)
        if not m_norm:
            continue
        sim = SequenceMatcher(None, c_norm, m_norm).ratio()
        neg_c = any(w in f" {c_norm} " for w in _NEG_WORDS)
        neg_m = any(w in f" {m_norm} " for w in _NEG_WORDS)
        tension = 0.0
        if sim > 0.55 and neg_c != neg_m:
            tension = min(1.0, sim)
        elif sim > 0.72:
            # similar phrasing; small tension for merge review
            tension = min(1.0, 0.35 + 0.5 * (sim - 0.72))
        if tension >= 0.35:
            out.append(
                ContradictionSignal(
                    existing_memory_id=mem.id,
                    new_text=candidate_text,
                    existing_text=mem.content,
                    score=tension,
                ),
            )
    return out


def links_from_contradictions(
    new_memory_id: UUID,
    signals: list[ContradictionSignal],
    *,
    threshold: float = 0.55,
) -> list[LinkSuggestion]:
    """Propose ``CONTRADICTS`` edges for high-tension pairs."""
    links: list[LinkSuggestion] = []
    for s in signals:
        if s.score >= threshold:
            links.append(
                LinkSuggestion(
                    source_id=new_memory_id,
                    target_id=s.existing_memory_id,
                    relation=GraphRelationType.CONTRADICTS,
                    weight=min(1.0, s.score),
                ),
            )
    return links


def edge_from_suggestion(suggestion: LinkSuggestion) -> MemoryEdge:
    """Materialize a graph edge model from a link suggestion."""
    return MemoryEdge(
        source_id=suggestion.source_id,
        target_id=suggestion.target_id,
        relation=suggestion.relation,
        weight=suggestion.weight,
    )

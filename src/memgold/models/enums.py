"""Enumerations for the memory domain."""

from __future__ import annotations

from enum import Enum


class MemoryType(str, Enum):
    """Taxonomy aligned with cognitive memory systems."""

    SHORT_TERM = "short_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class GraphRelationType(str, Enum):
    """Lightweight typed edges between memories and events."""

    RELATED_TO = "related_to"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    SUPPORTS = "supports"
    SUPERSEDES = "supersedes"


class MemorySource(str, Enum):
    """Provenance coarse labels."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    SYSTEM_EVENT = "system_event"
    CONSOLIDATION = "consolidation"
    REFLECTION = "reflection"
    IMPORT = "import"
    UNKNOWN = "unknown"

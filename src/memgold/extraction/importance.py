"""Importance / salience heuristics for memory prioritization."""

from __future__ import annotations

import math
import re

from memgold.models.extraction import MemoryCandidate


def score_importance(candidate: MemoryCandidate, *, text_len: int) -> float:
    """Combine extractor confidence, lexical density, and length into salience.

    TODO: Replace with learned utility model or LLM self-reflection scores.
    """
    base = float(candidate.confidence)
    kw_boost = min(0.25, 0.02 * len(candidate.keywords))
    ent_boost = min(0.2, 0.03 * len(candidate.entities))
    len_boost = min(0.15, math.log1p(max(text_len, 1)) / 40.0)
    caps = len(re.findall(r"[A-Z]{2,}", candidate.content))
    noise_penalty = min(0.1, 0.002 * caps)
    return float(max(0.0, min(1.0, base + kw_boost + ent_boost + len_boost - noise_penalty)))

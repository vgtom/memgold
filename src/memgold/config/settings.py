"""Central configuration for the memory subsystem (no extra deps beyond stdlib)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class MemoryLayerSettings:
    """Tunable parameters for retrieval, ranking, decay, and working memory."""

    default_top_k: int = field(default_factory=lambda: _env_int("MEMGOLD_DEFAULT_TOP_K", 12))
    working_memory_max_messages: int = field(
        default_factory=lambda: _env_int("MEMGOLD_WORKING_MEMORY_MAX", 32),
    )
    working_memory_max_tokens_estimate: int = field(
        default_factory=lambda: _env_int("MEMGOLD_WORKING_MEMORY_MAX_TOKENS", 4096),
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("MEMGOLD_EMBED_BATCH", 32),
    )
    decay_half_life_days: float = field(
        default_factory=lambda: _env_float("MEMGOLD_DECAY_HALF_LIFE_DAYS", 30.0),
    )
    archive_decay_threshold: float = field(
        default_factory=lambda: _env_float("MEMGOLD_ARCHIVE_DECAY_THRESHOLD", 0.08),
    )
    enable_graph_expansion: bool = field(
        default_factory=lambda: _env_bool("MEMGOLD_GRAPH_EXPANSION", True),
    )
    graph_expansion_hops: int = field(
        default_factory=lambda: _env_int("MEMGOLD_GRAPH_HOPS", 1),
    )
    log_json: bool = field(default_factory=lambda: _env_bool("MEMGOLD_LOG_JSON", False))


def load_settings() -> MemoryLayerSettings:
    """Load settings from environment with safe defaults."""
    return MemoryLayerSettings()

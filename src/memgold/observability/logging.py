"""Lightweight structured logging hooks (stdlib only)."""

from __future__ import annotations

import json
import logging
from typing import Any

from memgold.config.settings import MemoryLayerSettings

_LOG = logging.getLogger("memgold")


def configure_logging(settings: MemoryLayerSettings) -> None:
    """Idempotently configure root handler for memgold if unset."""
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO)
    _LOG.setLevel(logging.INFO)


def log_event(name: str, fields: dict[str, Any], *, settings: MemoryLayerSettings) -> None:
    """Emit an observability event (JSON or key=value)."""
    if settings.log_json:
        _LOG.info("%s %s", name, json.dumps(fields, default=str))
    else:
        parts = " ".join(f"{k}={fields[k]!r}" for k in sorted(fields))
        _LOG.info("%s %s", name, parts)

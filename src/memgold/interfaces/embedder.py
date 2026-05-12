"""Embedding provider port."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Maps text to dense vectors; implementations may batch or call remote APIs."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector size produced by this embedder."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts preserving order."""

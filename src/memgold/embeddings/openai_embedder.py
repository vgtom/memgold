"""OpenAI-compatible embedding provider (optional ``httpx`` dependency).

Install: ``pip install httpx`` and set ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import os
from typing import Any

from memgold.interfaces.embedder import Embedder


class OpenAICompatibleEmbedder(Embedder):
    """Call OpenAI ``/v1/embeddings`` (or compatible base URL).

    TODO: Add retries, rate limiting, and structured logging hooks.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimensions: int | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dims = dimensions

    @property
    def dimensions(self) -> int:
        if self._dims is not None:
            return self._dims
        return 1536

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional path
            raise RuntimeError(
                "OpenAICompatibleEmbedder requires httpx; install memgold with openai extra.",
            ) from exc
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        payload: dict[str, Any] = {"model": self._model, "input": texts}
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{self._base_url}/embeddings", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        items = sorted(data["data"], key=lambda d: d["index"])
        return [it["embedding"] for it in items]

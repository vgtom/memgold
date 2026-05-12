"""Deterministic pseudo-embeddings for development (no external models)."""

from __future__ import annotations

import hashlib
import math

from memgold.interfaces.embedder import Embedder

_EMBED_DIM = 64


def stub_embed(text: str) -> list[float]:
    """Return a fixed-length vector derived from *text* (not semantically meaningful).

    The mapping is stable for the same input so retrieval behaves reproducibly
    in tests without calling an embedding API.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    need = 4 * _EMBED_DIM
    buf = (digest * ((need // len(digest)) + 1))[:need]
    out: list[float] = []
    for i in range(0, need, 4):
        chunk = buf[i : i + 4]
        val = int.from_bytes(chunk, "big") / float(2**32) * 2.0 - 1.0
        out.append(max(-1.0, min(1.0, val)))
    norm = math.sqrt(sum(v * v for v in out)) or 1.0
    return [v / norm for v in out]


class StubEmbedder(Embedder):
    """Local embedder with no network calls; implements :class:`Embedder`."""

    @property
    def dimensions(self) -> int:
        return _EMBED_DIM

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [stub_embed(t) for t in texts]

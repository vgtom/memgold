"""Local sentence-transformers embedder (optional dependency).

Install: ``pip install sentence-transformers torch``.
"""

from __future__ import annotations

from memgold.interfaces.embedder import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Wrap ``sentence_transformers.SentenceTransformer.encode`` (sync) in async API.

    TODO: Offload CPU/GPU work to a process pool for large batches.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None

    def _ensure(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "SentenceTransformersEmbedder requires sentence-transformers.",
                ) from exc
            self._model = SentenceTransformer(self._model_name)

    @property
    def dimensions(self) -> int:
        self._ensure()
        assert self._model is not None
        return int(self._model.get_sentence_embedding_dimension())

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self._ensure()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

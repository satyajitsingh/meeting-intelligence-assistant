"""Local embedding provider backed by fastembed (ONNX, no network at query time).

Chosen over a hosted embedding API so the project runs with a single API key
and no second vendor account. The model downloads once on first use and is
cached on disk by fastembed.
"""

import asyncio
from typing import TYPE_CHECKING, Any

from app.adapters.embeddings.base import EmbeddingError, ensure_dimension

if TYPE_CHECKING:  # pragma: no cover - import cost is avoided at runtime
    from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384


class FastEmbedProvider:
    """Embeds text with a locally executed sentence-transformer ONNX model.

    The model is created on first use rather than in ``__init__`` so that
    constructing the provider -- during application wiring, or in a test that
    never embeds anything -- does not trigger a download.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: TextEmbedding | None = None
        self._dimension = self._resolve_dimension(model_name)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = await asyncio.to_thread(self._embed_documents_sync, texts)
        ensure_dimension(vectors, self._dimension, provider=self._model_name)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vectors = await asyncio.to_thread(self._embed_query_sync, text)
        ensure_dimension(vectors, self._dimension, provider=self._model_name)
        return vectors[0]

    # -- blocking work, run in a worker thread ------------------------------

    def _embed_documents_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        try:
            return [self._to_list(vector) for vector in model.embed(texts)]
        except Exception as exc:  # provider failures are opaque
            raise EmbeddingError(
                "Failed to embed documents.",
                details={"model": self._model_name, "count": len(texts)},
            ) from exc

    def _embed_query_sync(self, text: str) -> list[list[float]]:
        model = self._ensure_model()
        try:
            # BGE models are trained asymmetrically: queries get their own
            # instruction prefix, which `query_embed` applies.
            return [self._to_list(vector) for vector in model.query_embed(text)]
        except Exception as exc:  # provider failures are opaque
            raise EmbeddingError(
                "Failed to embed query.",
                details={"model": self._model_name},
            ) from exc

    def _ensure_model(self) -> "TextEmbedding":
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise EmbeddingError("fastembed is not installed.") from exc

            try:
                self._model = TextEmbedding(model_name=self._model_name, cache_dir=self._cache_dir)
            except Exception as exc:  # download or runtime failure
                raise EmbeddingError(
                    "Failed to load the embedding model.",
                    details={"model": self._model_name},
                ) from exc

        return self._model

    @staticmethod
    def _to_list(vector: Any) -> list[float]:
        return [float(value) for value in vector]

    @staticmethod
    def _resolve_dimension(model_name: str) -> int:
        """Read the model's dimension from fastembed's static metadata.

        This is a local lookup table, not a download, so it stays cheap enough
        to run during construction.
        """
        try:
            from fastembed import TextEmbedding

            for description in TextEmbedding.list_supported_models():
                if description.get("model") == model_name:
                    return int(description["dim"])
        except Exception:  # fall back to the documented default
            return DEFAULT_DIMENSION

        return DEFAULT_DIMENSION

"""Deterministic offline embedding provider for tests.

Uses feature hashing rather than hashing the whole string: each word token is
hashed into a bucket and accumulated, so texts sharing vocabulary produce
similar vectors. That makes ranking assertions in vector-store and retrieval
tests meaningful, which a random-vector-per-string fake could not do.

``hashlib`` is used deliberately -- Python's built-in ``hash()`` is seeded per
process, so it would produce different vectors on every run.
"""

import hashlib
import re

import numpy as np

from app.adapters.embeddings.base import EmbeddingError

DEFAULT_DIMENSION = 64

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


class FakeEmbeddingProvider:
    """A stable, network-free :class:`EmbeddingProvider` implementation."""

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        if dimension <= 0:
            raise EmbeddingError(
                "dimension must be greater than zero.",
                details={"dimension": dimension},
            )
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Hash tokens into buckets, then L2-normalise.

        Text with no word tokens (empty or punctuation-only) yields the zero
        vector, which downstream code treats as "similar to nothing" rather
        than as an error.
        """
        vector = np.zeros(self._dimension, dtype=np.float64)

        for token in _TOKEN_PATTERN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dimension
            sign = 1.0 if int.from_bytes(digest[8:], "big") % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm

        return [float(value) for value in vector]

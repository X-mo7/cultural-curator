"""Embedding interface, deterministic fixtures, and a Mistral adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from typing import TYPE_CHECKING, Protocol

from cultural_curator.llm.client import (
    AsyncCallPolicy,
    MetricsObserver,
    call_async,
    sdk_client,
)

if TYPE_CHECKING:
    from mistralai.client import Mistral
    from mistralai.client.models import EmbeddingResponse


EmbeddingVector = list[float]


class AsyncTextEmbedder(Protocol):
    """One awaitable contract for provider, demo, and fixture embeddings."""

    async def embed_async(self, texts: list[str]) -> list[EmbeddingVector]: ...


class FixtureEmbedder:
    """Deterministic fixture-backed embedder for tests and local development."""

    def __init__(self, fixtures: Mapping[str, Sequence[float]]) -> None:
        self._fixtures = {
            text: [float(value) for value in vector]
            for text, vector in fixtures.items()
        }

    async def embed_async(self, texts: list[str]) -> list[EmbeddingVector]:
        missing = [text for text in texts if text not in self._fixtures]
        if missing:
            missing_text = ", ".join(repr(text) for text in missing)
            raise KeyError(f"no fixture embedding for: {missing_text}")

        return [list(self._fixtures[text]) for text in texts]


class MistralTextEmbedder:
    """Embed a batch with Mistral, preserving the input order.

    An injected client belongs to the caller. Otherwise, each non-empty call
    opens and closes its own async SDK client. Calls use bounded retries for
    transient failures, timeouts, and optional payload-free usage metrics.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "mistral-embed",
        *,
        client: Mistral | None = None,
        timeout_ms: int = 30_000,
        max_attempts: int = 3,
        observer: MetricsObserver | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be blank")
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self.model = model
        self.timeout_ms = timeout_ms
        self._policy = AsyncCallPolicy(timeout_ms=timeout_ms, max_attempts=max_attempts)
        self._observer = observer
        self._client = client
        self._api_key = None
        if client is None:
            self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
            if not self._api_key or not self._api_key.strip():
                raise RuntimeError("MISTRAL_API_KEY is required")

    async def embed_async(self, texts: list[str]) -> list[EmbeddingVector]:
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("embedding inputs must be non-blank strings")

        async with sdk_client(self._api_key, self._client) as client:
            return await call_async(
                lambda: client.embeddings.create_async(
                    model=self.model,
                    inputs=texts,
                    encoding_format="float",
                    timeout_ms=self.timeout_ms,
                    retries=None,
                ),
                operation="embeddings",
                model=self.model,
                policy=self._policy,
                observer=self._observer,
                validate=lambda response: _ordered_embeddings(response, len(texts)),
            )


def _ordered_embeddings(
    response: EmbeddingResponse, expected_count: int
) -> list[EmbeddingVector]:
    """Validate the SDK response before the ranker associates vectors with texts."""
    if len(response.data) != expected_count:
        raise ValueError("Mistral returned a different number of embeddings than texts")

    by_index: dict[int, EmbeddingVector] = {}
    dimension = None
    for item in response.data:
        index = item.index
        if type(index) is not int or not 0 <= index < expected_count or index in by_index:
            raise ValueError("Mistral returned missing, duplicate, or invalid embedding indices")

        vector = item.embedding
        if not vector:
            raise ValueError("Mistral returned a missing or empty embedding")
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("Mistral embeddings must contain only finite values")
        if not any(vector):
            raise ValueError("Mistral embeddings must be non-zero for cosine similarity")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError("Mistral embeddings must have the same dimension")
        by_index[index] = list(vector)

    return [by_index[index] for index in range(expected_count)]

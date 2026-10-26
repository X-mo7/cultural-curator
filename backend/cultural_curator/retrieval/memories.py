"""Small-corpus semantic retrieval followed by optional lexical BM25 reranking."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import re
from typing import TYPE_CHECKING
import unicodedata

from pydantic import Field

from cultural_curator.domain import CulturalMemory
from cultural_curator.domain.models import DomainModel
from cultural_curator.ranking.semantic import cosine_similarity

if TYPE_CHECKING:
    from cultural_curator.retrieval.embeddings import AsyncTextEmbedder


class RetrievedMemory(DomainModel):
    memory_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)
    rerank_score: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


def _tokens(text: str) -> list[str]:
    # Case/accent folding helps both French and English; there is no stemming.
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.findall(r"\w+", normalized)


def bm25_rerank(
    question: str, candidates: Sequence[RetrievedMemory]
) -> list[RetrievedMemory]:
    """Compute BM25 within the candidate pool, breaking ties by cosine then ID.

    BM25 rewards exact query terms, with diminishing returns for repeated terms
    and a document-length penalty. k1=1.5 and b=0.75 are fixed explicitly here.
    Its scores are not probabilities and are not mixed with cosine scores.
    """
    if not candidates:
        return []
    query_terms = set(_tokens(question))
    documents = [Counter(_tokens(candidate.text)) for candidate in candidates]
    document_lengths = [sum(document.values()) for document in documents]
    average_length = sum(document_lengths) / len(documents) or 1.0
    frequencies = {
        term: sum(term in document for document in documents) for term in query_terms
    }
    k1, b = 1.5, 0.75
    rescored = []
    for candidate, document, length in zip(
        candidates, documents, document_lengths, strict=True
    ):
        score = 0.0
        for term in sorted(query_terms):
            frequency = document[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5)
            )
            length_penalty = k1 * (1.0 - b + b * length / average_length)
            score += inverse_frequency * frequency * (k1 + 1.0) / (frequency + length_penalty)
        rescored.append(candidate.model_copy(update={"rerank_score": score}))
    return sorted(
        rescored,
        key=lambda item: (-item.rerank_score, -item.score, item.memory_id),
    )


def _validate_options(top_k: int, min_similarity: float, candidate_multiplier: int) -> None:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not math.isfinite(min_similarity) or not -1.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity must be finite and between -1 and 1")
    if type(candidate_multiplier) is not int or candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be a positive integer")


def _validate_inputs(question: str, memories: Sequence[CulturalMemory]) -> None:
    if not question.strip():
        raise ValueError("question must not be blank")
    if len({memory.id for memory in memories}) != len(memories):
        raise ValueError("memory IDs must be unique")


def retrieve_memories(
    question: str,
    memories: Sequence[CulturalMemory],
    query_vector: Sequence[float],
    memory_vectors: Sequence[Sequence[float]],
    *,
    top_k: int = 5,
    rerank: bool = True,
    min_similarity: float = 0.25,
    candidate_multiplier: int = 4,
) -> list[RetrievedMemory]:
    """Pure retrieval over precomputed vectors, also used by offline evaluation.

    The semantic threshold is applied before reranking. BM25 can reorder only
    the best top_k * candidate_multiplier semantic candidates, then we keep
    top_k. This is a lexical second stage, not a neural cross-encoder.
    """
    _validate_options(top_k, min_similarity, candidate_multiplier)
    _validate_inputs(question, memories)
    if len(memories) != len(memory_vectors):
        raise ValueError("memory vector count must equal memory count")
    candidates = []
    for memory, vector in zip(memories, memory_vectors, strict=True):
        score = cosine_similarity(query_vector, vector)
        if score >= min_similarity:
            candidates.append(
                RetrievedMemory(memory_id=memory.id, text=memory.text, score=score)
            )
    candidates.sort(key=lambda item: (-item.score, item.memory_id))
    if rerank:
        candidates = bm25_rerank(question, candidates[: top_k * candidate_multiplier])
    return candidates[:top_k]


class MemoryRetriever:
    """Embed one question plus its small memory corpus, then run pure retrieval.

    The caller loads memories from storage. This deliberately re-embeds the
    corpus per request: simple and stateless, but unsuitable for a large corpus.
    min_similarity is an explicit heuristic requiring calibration on real data.
    """

    def __init__(
        self,
        embedder: AsyncTextEmbedder,
        *,
        min_similarity: float = 0.25,
        candidate_multiplier: int = 4,
    ) -> None:
        _validate_options(1, min_similarity, candidate_multiplier)
        self.embedder = embedder
        self.min_similarity = min_similarity
        self.candidate_multiplier = candidate_multiplier

    async def search(
        self,
        question: str,
        memories: Sequence[CulturalMemory],
        *,
        top_k: int = 5,
        rerank: bool = True,
    ) -> list[RetrievedMemory]:
        _validate_options(top_k, self.min_similarity, self.candidate_multiplier)
        _validate_inputs(question, memories)
        if not memories:
            return []
        texts = [question, *(memory.text for memory in memories)]
        vectors = await self.embedder.embed_async(texts)
        if len(vectors) != len(texts):
            raise ValueError("embedder returned a different number of vectors than texts")
        return retrieve_memories(
            question,
            memories,
            vectors[0],
            vectors[1:],
            top_k=top_k,
            rerank=rerank,
            min_similarity=self.min_similarity,
            candidate_multiplier=self.candidate_multiplier,
        )

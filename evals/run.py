"""Run a deterministic, network-free retrieval/ranking/grounding evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence, Set
import json
import math
from statistics import mean

from pydantic import ValidationError

from cultural_curator.llm.rag import (
    ABSTENTION_ANSWER,
    ContextMemory,
    InvalidCitationError,
    RagAnswer,
    RagService,
)
from cultural_curator.ranking import rank_events_by_preferences_async
from cultural_curator.retrieval.memories import MemoryRetriever
from evals.fixtures import (
    EVENTS,
    EVENT_QRELS,
    MEMORIES,
    MEMORY_QRELS,
    MEMORY_QUERIES,
    PERSONAS,
    event_embedder,
    memory_embedder,
)


def _validate_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of relevant IDs found in the first k results; empty qrels give 0.

    Queries with no relevant documents must be evaluated for abstention
    separately, rather than counted as successes in macro recall.
    """
    _validate_k(k)
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, int], k: int) -> float:
    """Exponential gain (2**grade - 1), log2 discount, unknown IDs graded zero."""
    _validate_k(k)
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("ranked IDs must be unique")
    if any(type(grade) is not int or grade < 0 for grade in relevance.values()):
        raise ValueError("relevance grades must be non-negative integers")

    def dcg(grades: Sequence[int]) -> float:
        return sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades))

    ideal = dcg(sorted(relevance.values(), reverse=True)[:k])
    actual = dcg([relevance.get(identifier, 0) for identifier in ranked_ids[:k]])
    return actual / ideal if ideal else 0.0


async def evaluate_retrieval() -> dict:
    retriever = MemoryRetriever(memory_embedder())
    report = {}
    for name, rerank in (("baseline", False), ("bm25", True)):
        cases = {}
        for case_id, question in MEMORY_QUERIES.items():
            case = {}
            for k in (1, 2):
                ranked = await retriever.search(
                    question, MEMORIES, top_k=k, rerank=rerank,
                )
                identifiers = [item.memory_id for item in ranked]
                case[f"recall@{k}"] = recall_at_k(identifiers, MEMORY_QRELS[case_id], k)
                case[f"ids@{k}"] = identifiers
            cases[case_id] = case
        report[name] = {
            **{f"recall@{k}": mean(case[f"recall@{k}"] for case in cases.values()) for k in (1, 2)},
            "cases": cases,
        }
    return report


async def evaluate_ranking() -> dict:
    cases = {}
    for persona, preferences in PERSONAS.items():
        ranked = await rank_events_by_preferences_async(preferences, EVENTS, event_embedder())
        identifiers = [item.event_id for item in ranked]
        cases[persona] = {
            "ndcg@3": ndcg_at_k(identifiers, EVENT_QRELS[persona], 3),
            "event_ids": identifiers[:3],
        }
    return {"ndcg@3": mean(case["ndcg@3"] for case in cases.values()), "cases": cases}


class ScriptedGenerator:
    """Return a controlled model payload through the real output schema."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    async def generate(self, question: str, context: tuple[ContextMemory, ...]) -> RagAnswer:
        self.calls += 1
        return RagAnswer.model_validate(self.payload)


async def evaluate_rag() -> dict:
    valid = {
        "answer": "You enjoyed street photography in Paris.",
        "citations": [{"memory_id": "photo", "quote": "loved the street photography"}],
    }
    cases = (
        ("grounded", "answered", MEMORIES, MEMORY_QUERIES["photo"], valid, 1),
        ("empty_corpus", "abstained", (), MEMORY_QUERIES["photo"], {}, 0),
        ("irrelevant_context", "abstained", MEMORIES, "astronomy", {}, 0),
        ("unknown_id", "invalid_citation", MEMORIES, MEMORY_QUERIES["photo"], {
            **valid, "citations": [{"memory_id": "invented", "quote": "street photography"}],
        }, 1),
        ("unselected_id", "invalid_citation", MEMORIES, MEMORY_QUERIES["photo"], {
            **valid, "citations": [{"memory_id": "painting", "quote": "vivid paintings"}],
        }, 1),
        ("invented_quote", "invalid_citation", MEMORIES, MEMORY_QUERIES["photo"], {
            **valid, "citations": [{"memory_id": "photo", "quote": "I hated photography"}],
        }, 1),
        ("missing_citations", "invalid_response", MEMORIES, MEMORY_QUERIES["photo"], {
            "answer": "Unsupported answer without sources.",
        }, 1),
        ("model_abstention", "abstained", MEMORIES, MEMORY_QUERIES["photo"], {
            "answer": "Unsupported prose must not survive abstention.", "abstained": True,
        }, 1),
    )
    results = {}
    for name, expected, memories, question, payload, expected_calls in cases:
        generator = ScriptedGenerator(payload)
        service = RagService(MemoryRetriever(memory_embedder()), generator)
        canonical_abstention = True
        try:
            answer = await service.answer(question, memories, top_k=1)
            outcome = "abstained" if answer.abstained else "answered"
            if answer.abstained:
                canonical_abstention = answer.answer == ABSTENTION_ANSWER and not answer.citations
        except InvalidCitationError:
            outcome = "invalid_citation"
        except ValidationError:
            outcome = "invalid_response"
        results[name] = {
            "expected": expected,
            "outcome": outcome,
            "generator_calls": generator.calls,
            "passed": outcome == expected and generator.calls == expected_calls and canonical_abstention,
        }
    return {
        "passed": sum(case["passed"] for case in results.values()),
        "total": len(results),
        "cases": results,
    }


async def run_evaluation() -> dict:
    return {
        "scope": "Synthetic regression fixtures; not a measurement of live-model quality.",
        "retrieval": await evaluate_retrieval(),
        "event_ranking": await evaluate_ranking(),
        "rag": await evaluate_rag(),
    }


def main() -> None:
    report = asyncio.run(run_evaluation())
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if report["rag"]["passed"] != report["rag"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

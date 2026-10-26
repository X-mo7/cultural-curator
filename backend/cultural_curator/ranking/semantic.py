"""Transparent semantic ranking baseline for cultural events."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
from typing import TYPE_CHECKING

from cultural_curator.domain import (
    CulturalEvent,
    Preference,
    PreferencePolarity,
    RankedEvent,
)
from cultural_curator.domain.models import PreferenceContribution

if TYPE_CHECKING:
    from cultural_curator.retrieval.embeddings import AsyncTextEmbedder


def event_text(event: CulturalEvent) -> str:
    """Return the deterministic V0 text representation embedded for an event."""
    parts = [
        f"Title: {event.title.strip()}",
    ]
    if event.category and event.category.strip():
        parts.append(f"Category: {event.category.strip()}")
    parts.append(f"Description: {event.description.strip()}")
    return "\n".join(parts)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two valid non-zero vectors."""
    if not left or not right:
        raise ValueError("cosine similarity requires non-empty vectors")
    if len(left) != len(right):
        raise ValueError("cosine similarity requires vectors with the same dimension")

    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("cosine similarity requires finite numeric vectors") from error
    if any(not math.isfinite(value) for value in left_values + right_values):
        raise ValueError("cosine similarity requires finite vectors")

    left_scale = max(abs(value) for value in left_values)
    right_scale = max(abs(value) for value in right_values)
    if left_scale == 0.0 or right_scale == 0.0:
        raise ValueError("cosine similarity is undefined for zero vectors")

    # Scale before normalizing to avoid overflow or underflow for valid vectors.
    left_values = [value / left_scale for value in left_values]
    right_values = [value / right_scale for value in right_values]
    left_norm = math.hypot(*left_values)
    right_norm = math.hypot(*right_values)
    similarity = math.fsum(
        (left_value / left_norm) * (right_value / right_norm)
        for left_value, right_value in zip(left_values, right_values, strict=True)
    )
    return min(1.0, max(-1.0, similarity))


async def rank_events_by_preferences_async(
    preferences: Iterable[Preference],
    events: Iterable[CulturalEvent],
    embedder: AsyncTextEmbedder,
) -> list[RankedEvent]:
    """Embed one batch, then apply the synchronous deterministic scoring core."""
    preference_list = list(preferences)
    event_list = list(events)
    embeddings = (
        await embedder.embed_async(_embedding_texts(preference_list, event_list))
        if preference_list and event_list
        else []
    )
    return rank_events_from_embeddings(preference_list, event_list, embeddings)


def _embedding_texts(
    preferences: Sequence[Preference], events: Sequence[CulturalEvent]
) -> list[str]:
    return [preference.label for preference in preferences] + [event_text(event) for event in events]


def rank_events_from_embeddings(
    preferences: Sequence[Preference],
    events: Sequence[CulturalEvent],
    embeddings: Sequence[Sequence[float]],
) -> list[RankedEvent]:
    """Score vectors ordered as preferences then events, without external calls.

    The baseline remains mean positive cosine minus mean negative cosine.
    Every signed term is returned, including zero terms, so callers can inspect
    exactly which preferences were evaluated and reconstruct the raw score.
    Empty inputs require no embeddings; without preferences, ties use event ID.
    """
    if not preferences or not events:
        if embeddings:
            raise ValueError("no embeddings expected without preferences or events")
        return sorted(
            (
                RankedEvent(
                    event_id=event.id,
                    score=0.0,
                    preference_contributions=(),
                )
                for event in events
            ),
            key=lambda ranked_event: ranked_event.event_id,
        )

    if len(embeddings) != len(preferences) + len(events):
        raise ValueError("embedder returned a different number of vectors than texts")

    positive_count = sum(
        preference.polarity is PreferencePolarity.POSITIVE
        for preference in preferences
    )
    negative_count = len(preferences) - positive_count
    preference_vectors = embeddings[:len(preferences)]
    event_vectors = embeddings[len(preferences):]

    ranked_events = []
    for event, event_vector in zip(events, event_vectors, strict=True):
        contributions = []
        for preference, vector in zip(preferences, preference_vectors, strict=True):
            similarity = cosine_similarity(event_vector, vector)
            weight = (
                1.0 / positive_count
                if preference.polarity is PreferencePolarity.POSITIVE
                else -1.0 / negative_count
            )
            contributions.append(
                PreferenceContribution(
                    preference=preference,
                    cosine_similarity=similarity,
                    weight=weight,
                    contribution=similarity * weight,
                )
            )
        ranked_events.append(
            RankedEvent(
                event_id=event.id,
                score=math.fsum(item.contribution for item in contributions),
                preference_contributions=tuple(contributions),
            )
        )

    return sorted(
        ranked_events,
        key=lambda ranked_event: (-ranked_event.score, ranked_event.event_id),
    )

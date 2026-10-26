"""Minimal deterministic assembly of ranked events into recommendations."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice
from typing import TYPE_CHECKING

from cultural_curator.domain import (
    CulturalEvent,
    Preference,
    RankedEvent,
    Recommendation,
)
from cultural_curator.domain.models import PreferenceContribution
from cultural_curator.ranking.semantic import rank_events_by_preferences_async

if TYPE_CHECKING:
    from cultural_curator.retrieval.embeddings import AsyncTextEmbedder


RAW_RANKING_SCORE_MIN = -2.0
RAW_RANKING_SCORE_MAX = 2.0


async def recommend_events(
    preferences: Iterable[Preference],
    events: Iterable[CulturalEvent],
    embedder: AsyncTextEmbedder,
    *,
    top_k: int = 3,
) -> list[Recommendation]:
    """Embed, rank, and explain a catalogue without blocking external calls."""
    if top_k <= 0:
        return []
    preference_list = list(preferences)
    event_list = list(events)
    ranked_events = await rank_events_by_preferences_async( 
        preference_list, event_list, embedder
    ) #renvoie une liste de RankedEvent
    return assemble_recommendations(
        ranked_events, event_list, top_k=top_k
    )


def assemble_recommendations(
    ranked_events: Iterable[RankedEvent],
    events: Iterable[CulturalEvent],
    *,
    top_k: int = 3,
) -> list[Recommendation]:
    """Convert ranked events into grounded transient recommendations.

    Recommendation scores are normalized from the current semantic ranking
    formula's theoretical raw range, [-2, 2], into [0, 1]. The normalized score
    is only for presentation; it is not a calibrated probability or confidence
    estimate. New rankings carry the exact terms used to compute the score.
    Legacy rankings without attribution claim no supporting preference or memory.
    """
    if top_k <= 0:
        return []

    event_list = list(events)
    event_by_id = {event.id: event for event in event_list}
    if len(event_by_id) != len(event_list):
        raise ValueError("event IDs must be unique")
    recommendations = []

    for ranked_event in islice(ranked_events, top_k): #islice pour couper l'Iterable ranked_events à top_k éléments (particularité de islice : ne charge pas toutes les données en mémoire)
        event = event_by_id.get(ranked_event.event_id)
        if event is None:
            raise ValueError(f"ranked event {ranked_event.event_id!r} is missing")

        contributions = ranked_event.preference_contributions
        if contributions is None:
            evidence_preferences = ()
            explanation = (
                "Ranking attribution unavailable; no preference or memory "
                "is claimed as score evidence."
            )
        else:
            evidence_preferences = tuple(
                item.preference for item in contributions if item.contribution != 0.0
            )
            explanation = _explain_contributions(contributions)
        recommendations.append(
            Recommendation(
                event_id=ranked_event.event_id,
                event=event,
                ranking_score=ranked_event.score,
                score=_normalize_raw_score(ranked_event.score),
                preference_contributions=contributions,
                explanation=explanation,
                supporting_memory_ids=_supporting_memory_ids(evidence_preferences),
            )
        )

    return recommendations


def _explain_contributions(
    contributions: tuple[PreferenceContribution, ...],
) -> str:
    if not contributions:
        return "No stored preferences; no personalization was applied. Ties use event ID."
    nonzero = sorted(
        (item for item in contributions if item.contribution != 0.0),
        key=lambda item: (-abs(item.contribution), item.preference.id),
    )
    if not nonzero:
        return "All preference similarities are zero; ties use event ID."
    terms = [
        f'{item.preference.origin.value} {item.preference.polarity.value} preference '
        f'"{item.preference.label}": cosine {item.cosine_similarity:+.3f} '
        f"× weight {item.weight:+.3f} = {item.contribution:+.3f}"
        for item in nonzero
    ]
    return "Raw score contributions: " + "; ".join(terms) + "."


def _normalize_raw_score(score: float) -> float:
    normalized = (score - RAW_RANKING_SCORE_MIN) / (
        RAW_RANKING_SCORE_MAX - RAW_RANKING_SCORE_MIN
    )
    return min(1.0, max(0.0, normalized))


def _supporting_memory_ids(preferences: tuple[Preference, ...]) -> tuple[str, ...]:
    memory_ids = []
    seen = set()
    for preference in preferences:
        for memory_id in preference.source_memory_ids:
            if memory_id not in seen:
                memory_ids.append(memory_id)
                seen.add(memory_id)
    return tuple(memory_ids)

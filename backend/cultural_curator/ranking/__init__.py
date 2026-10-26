"""Recommendation ranking strategies."""

from cultural_curator.ranking.catalogue import rank_catalogue
from cultural_curator.ranking.recommendations import assemble_recommendations, recommend_events
from cultural_curator.ranking.semantic import (
    cosine_similarity,
    event_text,
    rank_events_from_embeddings,
    rank_events_by_preferences_async,
)

__all__ = [
    "assemble_recommendations",
    "cosine_similarity",
    "event_text",
    "rank_catalogue",
    "rank_events_from_embeddings",
    "rank_events_by_preferences_async",
    "recommend_events",
]

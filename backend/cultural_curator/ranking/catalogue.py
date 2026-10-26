"""Rank a stored catalogue through async repository and embedding contracts."""

from cultural_curator.domain import RankedEvent
from cultural_curator.persistence import CulturalEventRepository, PreferenceRepository
from cultural_curator.ranking.semantic import rank_events_by_preferences_async
from cultural_curator.retrieval.embeddings import AsyncTextEmbedder


async def rank_catalogue(
    preference_repository: PreferenceRepository,
    event_repository: CulturalEventRepository,
    embedder: AsyncTextEmbedder,
) -> list[RankedEvent]:
    preferences = await preference_repository.list_all()
    events = await event_repository.list_all()
    return await rank_events_by_preferences_async(preferences, events, embedder)

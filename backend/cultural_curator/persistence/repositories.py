"""Async storage contracts used by application services, independent of MongoDB."""

from typing import Protocol, TypeVar

from cultural_curator.domain import CulturalEvent, CulturalMemory, Preference
from cultural_curator.domain.models import DomainModel


ModelT = TypeVar("ModelT", bound=DomainModel)


class Repository(Protocol[ModelT]):
    async def add(self, item: ModelT) -> None:
        ...

    async def get(self, item_id: str) -> ModelT | None:
        ...

    async def list_all(self) -> list[ModelT]:
        ...


CulturalMemoryRepository = Repository[CulturalMemory]
PreferenceRepository = Repository[Preference]
CulturalEventRepository = Repository[CulturalEvent]

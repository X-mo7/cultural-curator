"""Application service for persisting memories and explicit preferences."""
import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from cultural_curator.domain import CulturalMemory, Preference, PreferenceOrigin
from cultural_curator.llm import PreferenceExtraction
from cultural_curator.llm.preference_extraction import AsyncPreferenceExtractor
from cultural_curator.persistence import CulturalMemoryRepository, PreferenceRepository
from cultural_curator.persistence.errors import PersistenceError


class UnsupportedEvidenceError(ValueError):
    """Raised when extracted evidence is absent from the source memory."""


@dataclass(frozen=True)
class IngestionResult:
    memory: CulturalMemory
    extraction: PreferenceExtraction # avec PreferenceExtraction : tuple[ExtractedPreference, ...]
    preferences: tuple[Preference, ...]


class IngestionFailure(RuntimeError):
    """Extraction failed, but the original memory has already been saved."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__("Memory saved; preference extraction failed")


class IngestionPersistenceFailure(PersistenceError):
    """Memory saved; preference writes failed and may be partially committed."""

    def __init__(self, memory_id: str) -> None:
        self.memory_id = memory_id
        super().__init__("Memory saved; preference storage failed")


class CulturalMemoryIngestionService:
    def __init__(
        self,
        memory_repository: CulturalMemoryRepository,
        preference_repository: PreferenceRepository,
        preference_extractor: AsyncPreferenceExtractor,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.memory_repository = memory_repository
        self.preference_repository = preference_repository
        self.preference_extractor = preference_extractor
        self.id_factory = id_factory or (lambda: str(uuid4()))
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    async def ingest_async(self, text: str) -> IngestionResult:
        """Await native async storage, then extraction, then preference writes."""
        extract_async = getattr(self.preference_extractor, "extract_async", None)
        if not callable(extract_async):
            raise TypeError(
                "Asynchronous ingestion requires a callable extract_async method"
            )
        memory = await self._save_memory(text)
        try:
            extraction = await extract_async(memory.text)
            return await self._save_preferences(memory, extraction)
        except PersistenceError as error:
            raise IngestionPersistenceFailure(memory.id) from error 
        except (RuntimeError, ValueError) as error:
            raise IngestionFailure(memory.id) from error

    async def ingest_batch(
        self, texts: list[str], *, concurrency: int = 3
    ) -> list[IngestionResult | BaseException]:
        """Independent memories run concurrently, with ordered per-item outcomes.

        A failed extraction does not cancel successful siblings. Cancellation of
        the batch itself still propagates. No retry ever re-inserts a memory.
        """
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        semaphore = asyncio.Semaphore(concurrency)

        async def ingest_one(text: str) -> IngestionResult:
            async with semaphore:
                return await self.ingest_async(text)

        return await asyncio.gather(
            *(ingest_one(text) for text in texts), return_exceptions=True
        )

    async def _save_memory(self, text: str) -> CulturalMemory:
        memory = CulturalMemory(
            id=self.id_factory(),
            text=text,
            created_at=self.clock(),
        )
        await self.memory_repository.add(memory)
        return memory

    async def _save_preferences(
        self, memory: CulturalMemory, extraction: PreferenceExtraction
    ) -> IngestionResult:
        unsupported = [
            item
            for item in extraction.preferences
            if item.evidence_quote not in memory.text
        ]
        if unsupported:
            labels = ", ".join(repr(item.label) for item in unsupported)
            raise UnsupportedEvidenceError(
                f"extracted preferences have unsupported evidence: {labels}"
            )

        preferences = tuple(
            Preference(
                id=self.id_factory(),
                label=item.label,
                polarity=item.polarity,
                origin=PreferenceOrigin.EXPLICIT,
                source_memory_ids=(memory.id,),
            )
            for item in extraction.preferences
        )
        for preference in preferences:
            await self.preference_repository.add(preference)

        return IngestionResult(
            memory=memory,
            extraction=extraction,
            preferences=preferences,
        )

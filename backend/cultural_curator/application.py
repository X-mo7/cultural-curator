"""Wire existing components together; keep HTTP and SDK details out of use cases."""

from dataclasses import dataclass

from cultural_curator.domain import Recommendation
from cultural_curator.ingestion.service import CulturalMemoryIngestionService
from cultural_curator.llm.assistant import ToolAssistant, ToolChat
from cultural_curator.llm.rag import RagAnswer, RagService
from cultural_curator.persistence import (
    CulturalEventRepository,
    CulturalMemoryRepository,
    PreferenceRepository,
)
from cultural_curator.ranking.recommendations import recommend_events
from cultural_curator.retrieval.embeddings import AsyncTextEmbedder
from cultural_curator.retrieval.memories import MemoryRetriever, RetrievedMemory


@dataclass
class Curator:
    ingestion: CulturalMemoryIngestionService
    memories: CulturalMemoryRepository
    preferences: PreferenceRepository
    events: CulturalEventRepository
    embedder: AsyncTextEmbedder
    retriever: MemoryRetriever
    rag: RagService
    chat: ToolChat

    async def search(
        self, question: str, *, top_k: int = 5, rerank: bool = True
    ) -> list[RetrievedMemory]:
        memories = await self.memories.list_all()
        return await self.retriever.search(
            question, memories, top_k=top_k, rerank=rerank
        )

    async def recommend(self, *, top_k: int = 3) -> list[Recommendation]:
        preferences = await self.preferences.list_all()
        events = await self.events.list_all()
        return await recommend_events(preferences, events, self.embedder, top_k=top_k)

    async def ask(
        self, question: str, *, top_k: int = 5, rerank: bool = True
    ) -> RagAnswer:
        memories = await self.memories.list_all()
        return await self.rag.answer(question, memories, top_k=top_k, rerank=rerank)

    def assistant(self) -> ToolAssistant:
        return ToolAssistant(self.chat, self.search, self.recommend)


def build_curator(
    *,
    memories: CulturalMemoryRepository,
    preferences: PreferenceRepository,
    events: CulturalEventRepository,
    demo: bool = False,
) -> Curator:
    """Build adapters without making a model call or loading personal records."""
    if demo:
        from cultural_curator.demo import (
            DemoChat,
            DemoEmbedder,
            DemoExtractor,
            DemoGenerator,
        )

        embedder = DemoEmbedder()
        extractor = DemoExtractor()
        chat = DemoChat()
        generator = DemoGenerator()
    else:
        from cultural_curator.llm.client import MistralChat
        from cultural_curator.llm.preference_extraction import (
            MistralPreferenceExtractor,
        )
        from cultural_curator.llm.rag import MistralRagGenerator
        from cultural_curator.retrieval.embeddings import MistralTextEmbedder

        embedder = MistralTextEmbedder()
        extractor = MistralPreferenceExtractor()
        chat = MistralChat()
        generator = MistralRagGenerator(chat)
    retriever = MemoryRetriever(embedder)
    return Curator(
        ingestion=CulturalMemoryIngestionService(memories, preferences, extractor),
        memories=memories,
        preferences=preferences,
        events=events,
        embedder=embedder,
        retriever=retriever,
        rag=RagService(retriever, generator),
        chat=chat,
    )

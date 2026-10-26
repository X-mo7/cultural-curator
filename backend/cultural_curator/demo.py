"""Reproducible offline doubles and an end-to-end HTTP scenario.

These deliberately limited rules exercise the plumbing, not Mistral's quality.
The CLI uses the configured MongoDB database; seed its catalogue first.
Each run appends observations and never deletes data.
"""

import asyncio
from hashlib import blake2b
import json
import re
from types import SimpleNamespace

from cultural_curator.llm.preference_extraction import (
    ExtractedPreference,
    PreferenceExtraction,
)
from cultural_curator.llm.rag import ContextMemory, RagAnswer, RagCitation


class DemoEmbedder:
    """Stable word hashing, explicitly a lexical offline stand-in for embeddings."""

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        # Match the I/O contract; this small offline fake needs no worker thread.
        return _hash_embeddings(texts)


def _hash_embeddings(texts: list[str]) -> list[list[float]]:
    vectors = []
    stopwords = {
        "i",
        "the",
        "a",
        "of",
        "my",
        "did",
        "what",
        "and",
        "in",
        "for",
        "de",
        "la",
        "le",
    }
    for text in texts:
        vector = [0.0] * 256
        for token in re.findall(r"\w+", text.casefold()):
            if token not in stopwords:
                index = (
                    int.from_bytes(blake2b(token.encode(), digest_size=4).digest())
                    % 256
                )
                vector[index] += 1
        if not any(vector):
            vector[0] = 1.0
        vectors.append(vector)
    return vectors


class DemoExtractor:
    async def extract_async(self, memory_text: str) -> PreferenceExtraction:
        preferences = []
        for match in re.finditer(
            r"\bI (loved|liked|enjoyed|disliked|hated) ([^.!?]+)", memory_text
        ):
            preferences.append(
                ExtractedPreference(
                    label=match[2],
                    polarity="negative"
                    if match[1] in {"disliked", "hated"}
                    else "positive",
                    evidence_quote=match[0],
                )
            )
        return PreferenceExtraction(preferences=tuple(preferences))


class DemoGenerator:
    async def generate(
        self, question: str, context: tuple[ContextMemory, ...]
    ) -> RagAnswer:
        source = context[0]
        return RagAnswer(
            answer=f"Souvenir retrouvé : {source.text}",
            citations=(RagCitation(memory_id=source.memory_id, quote=source.text),),
        )


class DemoChat:
    async def complete(self, *, messages: list[dict], tools: list[dict]):
        from mistralai.client.models import AssistantMessage, FunctionCall, ToolCall

        if messages[-1]["role"] == "user":
            message = AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="demo_search",
                        function=FunctionCall(
                            name="search_memories",
                            arguments=json.dumps(
                                {"question": "documentary photography", "top_k": 2}
                            ),
                        ),
                    ),
                    ToolCall(
                        id="demo_recommend",
                        function=FunctionCall(
                            name="recommend_events",
                            arguments='{"top_k": 2}',
                        ),
                    ),
                ],
            )
        else:
            message = AssistantMessage(
                content="Démonstration hors ligne : souvenirs recherchés et événements classés. Consultez les résultats des outils.",
                tool_calls=None,
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason="tool_calls" if message.tool_calls else "stop",
                )
            ]
        )


async def run_scenario() -> None:
    import httpx

    from cultural_curator.application import build_curator
    from cultural_curator.main import create_app

    from cultural_curator.persistence.mongodb import MongoSettings, open_store

    async with open_store(MongoSettings.from_env(mode="demo")) as store:
        curator = build_curator(
            demo=True,
            memories=store.memories,
            preferences=store.preferences,
            events=store.events,
        )
        transport = httpx.ASGITransport(app=create_app(curator))
        async with httpx.AsyncClient(
            transport=transport, base_url="http://demo"
        ) as client:
            scenario = [
                (
                    "/memories",
                    {"text": "I loved documentary photography of everyday urban life."},
                ),
                ("/memories", {"text": "I disliked conceptual installation."}),
                ("/recommendations", {"top_k": 2}),
                (
                    "/ask",
                    {
                        "question": "What documentary photography did I enjoy?",
                        "top_k": 2,
                    },
                ),
                ("/ask", {"question": "quantum computing superconductors"}),
                (
                    "/assistant",
                    {
                        "question": "Find my photography memories and recommend two events."
                    },
                ),
            ]
            for endpoint, payload in scenario:
                response = await client.post(endpoint, json=payload)
                response.raise_for_status()
                print(
                    json.dumps(
                        {"endpoint": endpoint, "result": response.json()},
                        ensure_ascii=False,
                        indent=2,
                    )
                )


def main() -> None:
    asyncio.run(run_scenario())


if __name__ == "__main__":
    main()

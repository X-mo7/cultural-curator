"""HTTP contracts and dependency injection; business logic lives in Curator."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from cultural_curator.application import Curator
from cultural_curator.domain import CulturalMemory, Preference, Recommendation
from cultural_curator.llm.assistant import AssistantAnswer, RecommendArguments, SearchArguments
from cultural_curator.llm.preference_extraction import PreferenceExtraction
from cultural_curator.llm.rag import RagAnswer


class MemoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(min_length=1, max_length=12000)

    @field_validator("text")
    @classmethod
    def require_nonblank(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("memory text must not be blank")
        return text  # Preserve original whitespace as part of the observation.


class MemoryResponse(BaseModel):
    memory: CulturalMemory
    extraction: PreferenceExtraction
    preferences: tuple[Preference, ...]


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)


def get_curator(request: Request) -> Curator:
    """Override this dependency, or pass a Curator to create_app, in tests."""
    return request.app.state.curator


Services = Annotated[Curator, Depends(get_curator)]
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/memories", response_model=MemoryResponse, status_code=201)
async def add_memory(body: MemoryRequest, curator: Services) -> MemoryResponse:
    result = await curator.ingestion.ingest_async(body.text)
    return MemoryResponse(memory=result.memory, extraction=result.extraction, preferences=result.preferences)


@router.post("/recommendations", response_model=list[Recommendation])
async def recommendations(body: RecommendArguments, curator: Services) -> list[Recommendation]:
    return await curator.recommend(top_k=body.top_k)


@router.post("/ask", response_model=RagAnswer)
async def ask(body: SearchArguments, curator: Services) -> RagAnswer:
    return await curator.ask(**body.model_dump())


@router.post("/assistant", response_model=AssistantAnswer)
async def assistant(body: AssistantRequest, curator: Services) -> AssistantAnswer:
    return await curator.assistant().answer(body.question)

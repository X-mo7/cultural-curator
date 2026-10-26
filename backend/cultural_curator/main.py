"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cultural_curator.api.routes import get_curator, router
from cultural_curator.application import Curator, build_curator
from cultural_curator.ingestion.service import (
    IngestionFailure,
    IngestionPersistenceFailure,
)
from cultural_curator.llm.assistant import ToolTurnLimitError
from cultural_curator.llm.client import ExternalCallError, InvalidModelResponse
from cultural_curator.llm.rag import InvalidCitationError
from cultural_curator.persistence.errors import DuplicateIdError, PersistenceError
from cultural_curator.persistence.mongodb import MongoSettings, open_store


def create_app(curator: Curator | None = None) -> FastAPI:
    """Create the Cultural Curator API application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if curator is None and get_curator not in app.dependency_overrides:
            mode = os.getenv("CULTURAL_CURATOR_MODE", "mistral")
            if mode not in {"demo", "mistral"}:
                raise ValueError("CULTURAL_CURATOR_MODE must be demo or mistral")
            async with open_store(MongoSettings.from_env()) as store:
                app.state.curator = build_curator(
                    demo=mode == "demo",
                    memories=store.memories,
                    preferences=store.preferences,
                    events=store.events,
                )
                try:
                    yield
                finally:
                    app.state.curator = None
        else:
            # Here we consider that injected dependencies will belong to the caller (test scenario).
            yield

    app = FastAPI(title="Cultural Curator API", version="0.1.0", lifespan=lifespan)
    app.state.curator = curator
    app.include_router(router)

    @app.exception_handler(PersistenceError)
    async def storage_failed(request: Request, error: PersistenceError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Storage operation failed; a write may have an unknown outcome.",
            },
        )

    @app.exception_handler(DuplicateIdError)
    async def duplicate_id(request: Request, error: DuplicateIdError) -> JSONResponse:
        return JSONResponse(
            status_code=409, content={"detail": "Application ID already exists."}
        )

    @app.exception_handler(IngestionPersistenceFailure)
    async def preference_storage_failed(
        request: Request, error: IngestionPersistenceFailure
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Memory saved; preference storage failed and may be partial. Do not resubmit the memory.",
                "memory_id": error.memory_id,
            },
        )

    @app.exception_handler(IngestionFailure)
    async def ingestion_failed(
        request: Request, error: IngestionFailure
    ) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={
                "detail": "Memory saved; extraction failed. Do not resubmit the memory.",
                "memory_id": error.memory_id,
            },
        )

    async def upstream_failed(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": "Model call failed or returned invalid output."},
        )

    for error_type in (
        ExternalCallError,
        InvalidModelResponse,
        InvalidCitationError,
        ToolTurnLimitError,
    ):
        app.add_exception_handler(error_type, upstream_failed)

    return app


app = create_app()

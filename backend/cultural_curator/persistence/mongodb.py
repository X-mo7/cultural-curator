"""Direct PyMongo async repositories and a single, explicit client lifecycle."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
import os
from typing import Any, Generic

from pydantic import ValidationError
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import DuplicateKeyError, PyMongoError

from cultural_curator.domain import CulturalEvent, CulturalMemory, Preference
from cultural_curator.persistence.errors import (
    DuplicateIdError,
    InvalidStoredDocumentError,
    PersistenceError,
)
from cultural_curator.persistence.repositories import ModelT


class MongoRepository(Generic[ModelT]):
    def __init__(
        self, collection: AsyncCollection[dict[str, Any]], model: type[ModelT]
    ) -> None:
        self.collection = collection
        self.model = model

    async def initialize(self) -> None:
        """MongoDB, rather than a read-before-write check, guarantees uniqueness."""
        try:
            await self.collection.create_index(
                "id", unique=True, name="unique_application_id"
            )
        except PyMongoError as error:
            raise PersistenceError(
                "Could not initialize the unique ID index"
            ) from error

    async def add(self, item: ModelT) -> None:
        # Revalidate even model_construct/model_copy inputs. JSON-compatible
        # BSON fields preserve offsets and microseconds (BSON Date loses both),
        # and encode enums, dates, URLs, tuples and nested models consistently.
        validated = self.model.model_validate(item.model_dump())
        document = validated.model_dump(mode="json")
        try:
            await self.collection.insert_one(document)
        except DuplicateKeyError as error:
            raise DuplicateIdError("Application ID already exists") from error
        except PyMongoError as error:
            raise PersistenceError(
                "Could not insert document; write outcome may be unknown"
            ) from error

    def _decode(self, document: dict[str, Any]) -> ModelT:
        # Remove only MongoDB metadata: unknown domain fields must still fail.
        fields = {key: value for key, value in document.items() if key != "_id"}
        try:
            return self.model.model_validate(fields)
        except ValidationError as error:
            raise InvalidStoredDocumentError(
                "Stored document failed domain validation"
            ) from error

    async def get(self, item_id: str) -> ModelT | None:
        try:
            document = await self.collection.find_one({"id": item_id})
        except PyMongoError as error:
            raise PersistenceError("Could not retrieve document") from error
        return self._decode(document) if document is not None else None

    async def list_all(self) -> list[ModelT]:
        try:
            # Explicit ordering avoids relying on MongoDB's natural order.
            # ObjectIds retain creation order within a client, like old appends.
            async with self.collection.find({}).sort("_id", 1) as cursor:
                return [self._decode(document) async for document in cursor]
        except PyMongoError as error:
            raise PersistenceError("Could not list documents") from error


@dataclass(frozen=True)
class MongoSettings:
    uri: str = "mongodb://127.0.0.1:27017"
    database: str = "cultural_curator"

    def __post_init__(self) -> None:
        if not self.uri.strip() or not self.database.strip():
            raise ValueError("MONGODB_URI and MONGODB_DATABASE must not be blank")

    @classmethod
    def from_env(cls, *, mode: str | None = None) -> "MongoSettings":
        mode = mode or os.getenv("CULTURAL_CURATOR_MODE", "mistral")
        default_database = (
            "cultural_curator_demo" if mode == "demo" else "cultural_curator"
        )
        return cls(
            uri=os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017"),
            database=os.getenv("MONGODB_DATABASE", default_database),
        )


@dataclass(frozen=True)
class MongoStore:
    memories: MongoRepository[CulturalMemory]
    preferences: MongoRepository[Preference]
    events: MongoRepository[CulturalEvent]


@asynccontextmanager
async def open_store(settings: MongoSettings) -> AsyncIterator[MongoStore]:
    """Create/use/close one client on one loop, including failed startup paths."""
    client: AsyncMongoClient[dict[str, Any]] = AsyncMongoClient(
        settings.uri,
        serverSelectionTimeoutMS=5000,
        timeoutMS=10000,
        tz_aware=True,
    )
    try:
        database = client[settings.database]
        store = MongoStore(
            MongoRepository(database["memories"], CulturalMemory),
            MongoRepository(database["preferences"], Preference),
            MongoRepository(database["events"], CulturalEvent),
        )
        try:
            await client.admin.command("ping")
        except PyMongoError as error:
            raise PersistenceError("Could not connect to MongoDB") from error
        for repository in (store.memories, store.preferences, store.events):
            await repository.initialize()
        yield store
    finally:
        await client.close()

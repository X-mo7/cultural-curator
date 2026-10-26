"""One-time JSONL import / catalogue seeding, never an application backend.

Validate without writes by default; --apply opts into inserts. Existing records
are never updated or deleted. Sources are never modified.
"""

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError

from cultural_curator.domain import CulturalEvent, CulturalMemory, Preference
from cultural_curator.persistence.errors import DuplicateIdError
from cultural_curator.persistence.mongodb import MongoSettings, MongoStore, open_store
from cultural_curator.persistence.repositories import ModelT, Repository


# Run from the repository root, or /app in the Docker image. The catalogue is
# external data, so its location must not depend on the installed package path.
SEED_CATALOGUE = Path("data/catalogues/synthetic_events.jsonl")


async def read_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    """Read off the event loop, then validate before any database operation."""
    text = await asyncio.to_thread(path.read_text, encoding="utf-8")
    return _parse_jsonl(text, model, path)


def _parse_jsonl(text: str, model: type[ModelT], path: Path) -> list[ModelT]:
    """Pure record validation with source locations for useful import errors."""
    records = []
    seen = set()
    for line_number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            item = model.model_validate_json(line)
        except ValidationError as error:
            raise ValueError(
                f"Invalid {model.__name__} at {path}:{line_number}"
            ) from error
        if item.id in seen:
            raise ValueError(f"Duplicate ID {item.id!r} at {path}:{line_number}")
        seen.add(item.id)
        records.append(item)
    return records


def _same_record(left: ModelT, right: ModelT) -> bool:
    # Compare serialized values too: datetime equality alone ignores offsets.
    return left.model_dump(mode="json") == right.model_dump(mode="json")


async def check_conflicts(
    repository: Repository[ModelT], records: list[ModelT]
) -> None:
    for record in records:
        existing = await repository.get(record.id)
        if existing is not None and not _same_record(existing, record):
            raise ValueError(
                f"Conflicting existing ID {record.id!r}; nothing will be overwritten"
            )


async def insert_records(
    repository: Repository[ModelT], records: list[ModelT]
) -> dict[str, int]:
    inserted = skipped = 0
    for record in records:
        try:
            await repository.add(record)
            inserted += 1
        except DuplicateIdError:
            # The unique index is authoritative, including concurrent importers.
            existing = await repository.get(record.id)
            if existing is None or not _same_record(existing, record):
                raise ValueError(
                    f"Conflicting existing ID {record.id!r}; import stopped"
                )
            skipped += 1
    return {"inserted": inserted, "identical_skipped": skipped}


async def import_records(
    store: MongoStore,
    memories: list[CulturalMemory],
    preferences: list[Preference],
    events: list[CulturalEvent],
) -> dict[str, dict[str, int]]:
    # Check all collections before the first write. This isn't a transaction:
    # interruption or a concurrent conflicting insert can leave a partial import.
    batches = (
        ("memories", store.memories, memories),
        ("preferences", store.preferences, preferences),
        ("events", store.events, events),
    )
    for _, repository, records in batches:
        await check_conflicts(repository, records)
    return {
        name: await insert_records(repository, records)
        for name, repository, records in batches
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memories", type=Path)
    parser.add_argument("--preferences", type=Path)
    catalogue = parser.add_mutually_exclusive_group()
    catalogue.add_argument("--events", type=Path)
    catalogue.add_argument("--seed-catalogue", action="store_true")
    parser.add_argument(
        "--apply", action="store_true", help="connect, initialize indexes and insert"
    )
    asyncio.run(run_import(parser.parse_args()))


async def run_import(args: argparse.Namespace) -> None:
    memories = await read_jsonl(args.memories, CulturalMemory) if args.memories else []
    preferences = await read_jsonl(args.preferences, Preference) if args.preferences else []
    events_path = SEED_CATALOGUE if args.seed_catalogue else args.events
    events = await read_jsonl(events_path, CulturalEvent) if events_path else []
    print(
        f"Validated: {len(memories)} memories, {len(preferences)} preferences, {len(events)} events"
    )
    if not args.apply:
        print(
            "Dry run: no database connection or writes. Add --apply to import / initialize indexes."
        )
        return

    settings = MongoSettings.from_env()
    print(f"Target database: {settings.database}")
    async with open_store(settings) as store:
        print(await import_records(store, memories, preferences, events))


if __name__ == "__main__":
    main()

"""Storage errors crossing the persistence boundary without driver types."""


class PersistenceError(RuntimeError):
    """A database operation failed; a failed write may have an unknown outcome."""


class DuplicateIdError(PersistenceError):
    """An application ID already exists in the collection."""


class InvalidStoredDocumentError(PersistenceError):
    """A stored document violates its domain model's contract."""

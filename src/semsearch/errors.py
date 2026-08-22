"""Exception hierarchy for semsearch (spec §7.7)."""


class SemSearchError(Exception):
    """Base class for all semsearch errors."""


class FileIngestError(SemSearchError):
    """A single file failed to ingest (provider down, bad format, etc.)."""


class SearchError(SemSearchError):
    """Embedding query failed or DB unreachable."""


class DeleteError(SemSearchError):
    """PGVectorStore.delete() raised (DB unreachable, etc.)."""


class SchemaMismatchError(SemSearchError):
    """Table exists but vector_size doesn't match expected dimension."""


class ProviderConfigError(SemSearchError):
    """Required credentials missing for the selected provider type."""

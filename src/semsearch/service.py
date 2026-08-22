"""Core service — SemanticSearchService facade (spec §7.6).

Phase D (PLAN.md): internals live in ``semsearch.services.*`` mixins
(base plumbing, ingest, search, admin). This module composes them and
preserves the exact public API — ``cli.py``, ``server.py``, tests and
scripts import ``SemanticSearchService`` from here unchanged.
"""

from semsearch.services.admin import AdminMixin
from semsearch.services.base import BaseService
from semsearch.services.ingest import SUPPORTED_EXTENSIONS, IngestMixin
from semsearch.services.search import SearchMixin

__all__ = ["SUPPORTED_EXTENSIONS", "SemanticSearchService"]


class SemanticSearchService(IngestMixin, SearchMixin, AdminMixin):
    """High-level facade over loader + splitter + embedder + PGVectorStore.

    Lifecycle::

        with SemanticSearchService.from_settings(settings) as svc:
            svc.init_schema()
            svc.ingest(...)
            svc.search(...)
            svc.delete(filter={...})
    """

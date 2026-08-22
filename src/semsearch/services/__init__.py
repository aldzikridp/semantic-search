"""Internal service mixins (PLAN.md Phase D).

Implementation detail of ``semsearch.service.SemanticSearchService`` — the
public entry point remains ``from semsearch.service import
SemanticSearchService``. This package's layout may change without notice.
"""

from .admin import AdminMixin
from .base import BaseService
from .ingest import IngestMixin
from .search import SearchMixin

__all__ = ["AdminMixin", "BaseService", "IngestMixin", "SearchMixin"]

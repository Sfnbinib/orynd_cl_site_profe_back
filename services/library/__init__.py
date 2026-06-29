"""Knowledge Library cluster — OPEN + CLOSED layers.

Public re-exports kept minimal; everything else stays in submodules to keep
import-time work light (Pydantic model graph is already substantial).
"""

from orynd_core.services.library.storage_abstract import (
    StorageBackend,
    ArticleSearchResult,
    StageMetrics,
)
from orynd_core.services.library.storage_factory import get_storage_backend

__all__ = [
    "StorageBackend",
    "ArticleSearchResult",
    "StageMetrics",
    "get_storage_backend",
]

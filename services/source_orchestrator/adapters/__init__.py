"""
Adapters package — one adapter per "access method" or per dedicated site.
"""
from .base import AdapterBase, SearchHit, AdapterError, AdapterTimeout, AdapterBlocked
from .generic_html import GenericHTMLAdapter
from .browser_harness import BrowserHarnessAdapter
from .duckduckgo_fallback import DuckDuckGoFallbackAdapter
from .dedicated_proxy import DedicatedProxyAdapter

__all__ = [
    "AdapterBase",
    "SearchHit",
    "AdapterError",
    "AdapterTimeout",
    "AdapterBlocked",
    "GenericHTMLAdapter",
    "BrowserHarnessAdapter",
    "DuckDuckGoFallbackAdapter",
    "DedicatedProxyAdapter",
]

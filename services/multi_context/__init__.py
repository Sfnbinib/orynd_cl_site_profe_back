"""Multi-Context UI backend — slash commands + context chips + event bus glue.

Per MULTI_CONTEXT_UI.md + HYBRID_WORKFLOW. Connects bottom chat (workflow)
↔ left agent panel (long tasks) ↔ scene canvas (3D selection) via a small
routing layer.

* Slash commands (/research /skill /mesh /library /attach /macro) → harness
* Context chips — selected 3D object → chip injected into bottom chat
"""

from orynd_core.services.multi_context.slash import (
    SLASH_COMMANDS,
    SlashResult,
    parse_slash,
)

__all__ = ["SLASH_COMMANDS", "SlashResult", "parse_slash"]

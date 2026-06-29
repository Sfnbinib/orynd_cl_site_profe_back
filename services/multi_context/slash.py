"""Slash command parser for bottom chat.

Founder UX (MULTI_CONTEXT_UI): bottom chat = workflow context. Slash
commands route to the right capability without leaving the chat.

Examples:
    /research yacht hull stability   → deep research (Phase 10)
    /skill mesh_decompose path=...   → invoke skill
    /macro create 20mm cube          → text→CoreOps
    /attach bearing                  → axes attachment suggest
    /library yacht                   → library search
    /mesh /path/to.stl               → mesh decompose

Each returns a SlashResult with a routing target the chat handler executes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

CommandTarget = Literal[
    "research", "skill", "macro", "attach", "library", "mesh", "help", "unknown"
]


@dataclass
class SlashResult:
    is_slash: bool
    command: Optional[str] = None
    target: CommandTarget = "unknown"
    args_text: str = ""
    route: dict = field(default_factory=dict)  # how chat handler should dispatch
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "is_slash": self.is_slash,
            "command": self.command,
            "target": self.target,
            "args_text": self.args_text,
            "route": dict(self.route),
            "error": self.error,
        }


# command → (target, description, route template)
SLASH_COMMANDS: dict[str, dict] = {
    "research": {
        "target": "research",
        "desc": "Deep research a topic → article in Library",
        "route": {"method": "POST", "path": "/research/light"},  # Phase 10 mock
    },
    "skill": {
        "target": "skill",
        "desc": "Invoke a skill: /skill <slug> key=val ...",
        "route": {"method": "POST", "path": "/skills/{slug}/invoke"},
    },
    "macro": {
        "target": "macro",
        "desc": "Natural language → CoreOps: /macro create 20mm cube",
        "route": {"method": "POST", "path": "/macro/parse"},
    },
    "attach": {
        "target": "attach",
        "desc": "Suggest catalog parts for selected axis: /attach bearing",
        "route": {"method": "POST", "path": "/attachment/suggest"},
    },
    "library": {
        "target": "library",
        "desc": "Search Knowledge Library: /library yacht hull",
        "route": {"method": "GET", "path": "/library/articles/search"},
    },
    "mesh": {
        "target": "mesh",
        "desc": "Decompose a mesh: /mesh /path/to.stl",
        "route": {"method": "POST", "path": "/skills/mesh_decompose/invoke"},
    },
    "help": {
        "target": "help",
        "desc": "List slash commands",
        "route": {},
    },
}


def parse_slash(text: str) -> SlashResult:
    """Parse a chat input. If it starts with '/', route to a command."""
    text = (text or "").strip()
    if not text.startswith("/"):
        return SlashResult(is_slash=False)

    body = text[1:].strip()
    if not body:
        return SlashResult(is_slash=True, command="", target="help",
                           route=SLASH_COMMANDS["help"]["route"])

    parts = body.split(None, 1)
    cmd = parts[0].lower()
    args_text = parts[1] if len(parts) > 1 else ""

    spec = SLASH_COMMANDS.get(cmd)
    if spec is None:
        return SlashResult(
            is_slash=True,
            command=cmd,
            target="unknown",
            args_text=args_text,
            error=f"unknown command /{cmd}. Try /help",
        )

    return SlashResult(
        is_slash=True,
        command=cmd,
        target=spec["target"],
        args_text=args_text,
        route=dict(spec["route"]),
    )


def help_text() -> list[dict]:
    return [
        {"command": f"/{name}", "description": spec["desc"]}
        for name, spec in SLASH_COMMANDS.items()
    ]


__all__ = ["CommandTarget", "SlashResult", "SLASH_COMMANDS", "parse_slash", "help_text"]

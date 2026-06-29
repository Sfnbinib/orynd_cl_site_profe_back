"""Detect existing local Ollama installation + installed models.

Founder explicit ask (FINAL_DECISIONS_2026-06-02 § H3): "проверка, которую
мы делаем, потому что если модель нужная установлена, то как бы да. Если
нет, мне придётся скачивать."

Returns a structured report so the installer can skip already-present
artifacts → saves bandwidth + disk space.

Defensive: gracefully handles "Ollama not installed / not running" so the
flow doesn't break.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import httpx

from orynd_core.services.connection_pool import get_ollama_client
from orynd_core.services.logging import get_logger

log = get_logger("orynd.installer.ollama_check")


@dataclass
class OllamaModel:
    name: str
    size_bytes: int = 0
    digest: str = ""
    modified_at: str = ""


@dataclass
class OllamaStatus:
    reachable: bool = False
    base_url: str = ""
    models: list[OllamaModel] = field(default_factory=list)
    error: Optional[str] = None

    def has_model(self, name: str) -> bool:
        """Match by exact name OR by name prefix (e.g. 'llama3.2' matches 'llama3.2:3b')."""
        for m in self.models:
            if m.name == name:
                return True
            # Allow checking by family — "llama3.2" matches "llama3.2:3b"
            base = m.name.split(":", 1)[0]
            if base == name:
                return True
        return False

    def total_local_bytes(self) -> int:
        return sum(m.size_bytes for m in self.models)

    def to_dict(self) -> dict:
        return {
            "reachable": self.reachable,
            "base_url": self.base_url,
            "error": self.error,
            "total_local_bytes": self.total_local_bytes(),
            "models": [asdict(m) for m in self.models],
        }


async def check_ollama() -> OllamaStatus:
    """Probe local Ollama via /api/tags. Never raises — always returns status."""
    client = get_ollama_client()
    base_url = str(client.base_url)
    try:
        resp = await client.get("/api/tags", timeout=2.0)
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return OllamaStatus(reachable=False, base_url=base_url, error=f"unreachable: {type(exc).__name__}")
    except Exception as exc:
        log.warning("ollama_check.unexpected", error=str(exc), exc_info=True)
        return OllamaStatus(reachable=False, base_url=base_url, error=f"error: {type(exc).__name__}")

    if resp.status_code != 200:
        return OllamaStatus(reachable=False, base_url=base_url, error=f"http {resp.status_code}")

    try:
        payload = resp.json()
    except ValueError:
        return OllamaStatus(reachable=True, base_url=base_url, error="bad json")

    raw_models = payload.get("models", []) if isinstance(payload, dict) else []
    models = [
        OllamaModel(
            name=str(m.get("name", "")),
            size_bytes=int(m.get("size", 0) or 0),
            digest=str(m.get("digest", "")),
            modified_at=str(m.get("modified_at", "")),
        )
        for m in raw_models
        if isinstance(m, dict) and m.get("name")
    ]
    return OllamaStatus(reachable=True, base_url=base_url, models=models)


def plan_download_skip(
    status: OllamaStatus,
    required_models: list[str],
) -> dict:
    """Decide which required models can be skipped.

    Returns:
        {
            "skip": [...names already installed],
            "download": [...names that must be downloaded],
            "bytes_saved": int,
        }
    """
    skip: list[str] = []
    download: list[str] = []
    bytes_saved = 0
    for name in required_models:
        if status.has_model(name):
            skip.append(name)
            for m in status.models:
                base = m.name.split(":", 1)[0]
                if m.name == name or base == name:
                    bytes_saved += m.size_bytes
                    break
        else:
            download.append(name)
    return {"skip": skip, "download": download, "bytes_saved": bytes_saved}


__all__ = ["OllamaModel", "OllamaStatus", "check_ollama", "plan_download_skip"]

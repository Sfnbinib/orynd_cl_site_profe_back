"""Drop 2 background installer.

Drop 1 (CAD + base) must complete during onboarding. Drop 2 (Ollama models,
full skills system, integration layer) installs in the background after the
user is already in the workspace.

Per FINAL_DECISIONS_2026-06-02 § H3:
* 3 retry attempts
* Drop 1 fail = onboarding blocked (handled elsewhere)
* Drop 2 fail = degraded mode + UI banner
* SHA-256 verification per artifact
* Resumable when possible
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx

from orynd_core.services.background_tasks import BackgroundTask, manager
from orynd_core.services.event_bus import bus
from orynd_core.services.logging import get_logger
from orynd_core.services.resilience.retry import network_retry

log = get_logger("orynd.installer.drop2")


class DropStatus(str, Enum):
    NOT_STARTED = "not_started"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    INSTALLING = "installing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Artifact:
    name: str
    url: str
    sha256: str
    install_path: Optional[Path] = None  # unpack/move target; None = leave in drop2 root
    size_bytes: int = 0


# ── Manifest ──────────────────────────────────────────────────────────────────
# The artifact list ships server-side so we can add/replace dependencies
# without releasing a new DMG. ORYND_DROP2_MANIFEST_URL overrides (e.g. GitHub
# raw / S3); the bundled drop2_manifest.json is the offline fallback.

_BUNDLED_MANIFEST = Path(__file__).parent / "drop2_manifest.json"


def _parse_manifest(data: dict) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for item in data.get("artifacts", []):
        artifacts.append(
            Artifact(
                name=str(item["name"]),
                url=str(item["url"]),
                sha256=str(item["sha256"]),
                install_path=Path(item["install_path"]).expanduser() if item.get("install_path") else None,
                size_bytes=int(item.get("size_bytes", 0) or 0),
            )
        )
    return artifacts


async def load_manifest() -> tuple[list[Artifact], str]:
    """Return (artifacts, source). Remote manifest wins; bundled is fallback."""
    url = os.environ.get("ORYND_DROP2_MANIFEST_URL", "").strip()
    if url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return _parse_manifest(resp.json()), "remote"
        except Exception as exc:
            log.warning("drop2.manifest_remote_failed", url=url, error=str(exc))
    try:
        data = json.loads(_BUNDLED_MANIFEST.read_text())
        return _parse_manifest(data), "bundled"
    except Exception as exc:
        log.warning("drop2.manifest_bundled_failed", error=str(exc))
        return [], "none"


@dataclass
class DropProgress:
    artifact: str
    bytes_downloaded: int = 0
    bytes_total: int = 0
    status: DropStatus = DropStatus.NOT_STARTED
    attempt: int = 0

    @property
    def fraction(self) -> float:
        if self.bytes_total == 0:
            return 0.0
        return self.bytes_downloaded / self.bytes_total


@dataclass
class DependencyDrop2Manager:
    """Coordinates downloading + verifying + installing Drop 2 artifacts."""

    artifacts: list[Artifact] = field(default_factory=list)
    max_retries: int = 3
    chunk_size: int = 64 * 1024
    install_root: Path = field(default_factory=lambda: Path(os.environ.get("ORYND_INSTALL_ROOT", "~/.orynd/drop2")).expanduser())
    _progress: dict[str, DropProgress] = field(default_factory=dict)

    async def start_background(
        self,
        artifacts: list[Artifact] | None = None,
        on_progress: Callable[[DropProgress], Awaitable[None]] | None = None,
    ) -> BackgroundTask:
        """Spawn the full Drop 2 flow as a tracked background task."""
        if artifacts:
            self.artifacts = artifacts

        async def runner(task: BackgroundTask) -> dict:
            return await self._run(task, on_progress)

        return manager.submit("drop2.install", runner)

    async def _run(
        self,
        task: BackgroundTask,
        on_progress: Callable[[DropProgress], Awaitable[None]] | None,
    ) -> dict:
        if not self.artifacts:
            log.info("drop2.skip_no_artifacts")
            return {"installed": [], "failed": []}

        self.install_root.mkdir(parents=True, exist_ok=True)
        installed, failed = [], []

        for index, artifact in enumerate(self.artifacts):
            progress = DropProgress(artifact=artifact.name, bytes_total=artifact.size_bytes)
            self._progress[artifact.name] = progress

            ok = await self._download_and_verify(artifact, progress, on_progress)
            if ok:
                installed.append(artifact.name)
            else:
                failed.append(artifact.name)
                await bus.publish(
                    "drop2.artifact_failed",
                    {"artifact": artifact.name, "attempts": progress.attempt},
                )

            await manager.update_progress(
                task,
                (index + 1) / len(self.artifacts),
                message=f"{artifact.name}: {progress.status.value}",
            )

        result = {"installed": installed, "failed": failed}
        await bus.publish("drop2.complete", result)
        return result

    async def _download_and_verify(
        self,
        artifact: Artifact,
        progress: DropProgress,
        on_progress: Callable[[DropProgress], Awaitable[None]] | None,
    ) -> bool:
        path = self.install_root / artifact.name
        for attempt in range(1, self.max_retries + 1):
            progress.attempt = attempt
            try:
                progress.status = DropStatus.DOWNLOADING
                await _emit(progress, on_progress)
                await self._download(artifact, path, progress, on_progress)

                progress.status = DropStatus.VERIFYING
                await _emit(progress, on_progress)
                if not _verify_sha256(path, artifact.sha256):
                    # Corrupt file must not survive into the resume logic.
                    path.unlink(missing_ok=True)
                    raise ValueError(f"sha256 mismatch for {artifact.name}")

                progress.status = DropStatus.INSTALLING
                await _emit(progress, on_progress)
                await asyncio.to_thread(_install_artifact, artifact, path)

                progress.status = DropStatus.COMPLETED
                await _emit(progress, on_progress)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "drop2.artifact_attempt_failed",
                    artifact=artifact.name,
                    attempt=attempt,
                    error=str(exc),
                )
                if attempt == self.max_retries:
                    progress.status = DropStatus.FAILED
                    await _emit(progress, on_progress)
                    return False
                # Exponential backoff before retry.
                await asyncio.sleep(min(30, 2**attempt))
        return False

    @network_retry
    async def _download(
        self,
        artifact: Artifact,
        dest: Path,
        progress: DropProgress,
        on_progress: Callable[[DropProgress], Awaitable[None]] | None,
    ) -> None:
        # Resume: when a partial file is already on disk, ask for the rest.
        existing = dest.stat().st_size if dest.exists() else 0
        headers: dict[str, str] = {}
        if existing > 0 and (artifact.size_bytes == 0 or existing < artifact.size_bytes):
            headers["Range"] = f"bytes={existing}-"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0),
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", artifact.url, headers=headers) as resp:
                resp.raise_for_status()
                resumed = resp.status_code == 206
                length = int(resp.headers.get("Content-Length") or 0)
                progress.bytes_total = (existing + length) if resumed else (
                    length or artifact.size_bytes or 0
                )
                progress.bytes_downloaded = existing if resumed else 0
                mode = "ab" if resumed else "wb"
                with open(dest, mode) as fh:
                    async for chunk in resp.aiter_bytes(self.chunk_size):
                        fh.write(chunk)
                        progress.bytes_downloaded += len(chunk)
                        await _emit(progress, on_progress)


def _safe_members(names: list[str], root: Path) -> None:
    """Reject archive members that would escape the extraction root."""
    for name in names:
        target = (root / name).resolve()
        if not str(target).startswith(str(root.resolve()) + os.sep) and target != root.resolve():
            raise ValueError(f"unsafe archive member: {name}")


def _install_artifact(artifact: Artifact, downloaded: Path) -> None:
    """Post-download install step.

    * .zip / .tar.gz / .tgz with install_path → safe-extract there
    * plain file with install_path → move there
    * no install_path → the file in the drop2 root *is* the install
    """
    target = artifact.install_path
    if target is None:
        return
    target = target.expanduser()
    name = downloaded.name.lower()

    if name.endswith(".zip"):
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(downloaded) as zf:
            _safe_members(zf.namelist(), target)
            zf.extractall(target)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(downloaded) as tf:
            _safe_members([m.name for m in tf.getmembers()], target)
            tf.extractall(target)
    else:
        if target.resolve() == downloaded.resolve():
            return  # already in place — install_path points at the drop2 root
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, target)


def _verify_sha256(path: Path, expected: str) -> bool:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(64 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == expected.lower()


async def _emit(progress: DropProgress, on_progress: Callable[[DropProgress], Awaitable[None]] | None) -> None:
    if on_progress is None:
        return
    try:
        await on_progress(progress)
    except Exception:
        pass


__all__ = [
    "Artifact",
    "DependencyDrop2Manager",
    "DropProgress",
    "DropStatus",
    "load_manifest",
]

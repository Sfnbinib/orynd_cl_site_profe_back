"""Hardware-ID computation.

HWID = ``sha256(machine_id + install_id)``.

* ``machine_id`` — OS-provided identifier (stable per machine reinstall).
* ``install_id`` — UUID generated on first ORYND launch, persisted under
  ``~/.orynd/install_id``. Stays even across reinstall (so users don't
  rebind every clean install) unless the user manually deletes the file.

The license server binds an issued JWT to this HWID. Verification at
runtime ensures the license isn't transplanted to a different machine.

Privacy: HWID is one-way hash — the server never learns the raw machine_id.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import uuid
from pathlib import Path
from typing import Optional


def _install_id_path() -> Path:
    base = Path(os.environ.get("ORYND_INSTALL_DIR", "~/.orynd")).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base / "install_id"


def get_or_create_install_id() -> str:
    path = _install_id_path()
    if path.exists():
        try:
            value = path.read_text().strip()
            if value:
                return value
        except OSError:
            pass
    new_id = uuid.uuid4().hex
    try:
        path.write_text(new_id)
    except OSError:
        # Read-only fs / sandbox — still return value for this process.
        pass
    return new_id


def _machine_id_macos() -> Optional[str]:
    """Read IOPlatformUUID via ioreg. Stable per Mac hardware."""
    try:
        out = subprocess.run(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if "IOPlatformUUID" in line:
            parts = line.split('"')
            if len(parts) >= 4:
                return parts[3]
    return None


def _machine_id_linux() -> Optional[str]:
    for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            text = Path(candidate).read_text().strip()
            if text:
                return text
        except OSError:
            continue
    return None


def _machine_id_windows() -> Optional[str]:
    try:
        out = subprocess.run(
            ["wmic", "csproduct", "get", "uuid"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned.lower() != "uuid":
            return cleaned
    return None


def get_machine_id() -> str:
    """OS-stable machine identifier. Falls back to platform.node() if unavailable."""
    override = os.environ.get("ORYND_MACHINE_ID_OVERRIDE")
    if override:
        return override
    system = platform.system().lower()
    if system == "darwin":
        v = _machine_id_macos()
        if v:
            return v
    elif system == "linux":
        v = _machine_id_linux()
        if v:
            return v
    elif system == "windows":
        v = _machine_id_windows()
        if v:
            return v
    return platform.node() or "unknown"


def compute_hwid(machine_id: Optional[str] = None, install_id: Optional[str] = None) -> str:
    """Return ``sha256(machine_id + ':' + install_id)`` as hex."""
    m = machine_id or get_machine_id()
    i = install_id or get_or_create_install_id()
    return hashlib.sha256(f"{m}:{i}".encode("utf-8")).hexdigest()


__all__ = ["compute_hwid", "get_machine_id", "get_or_create_install_id"]

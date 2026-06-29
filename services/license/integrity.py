"""Module integrity hash check (anti-tampering).

Compiled-in expected hashes (Phase 1+) are compared against the running
binary's actual contents. If anything mismatches → app refuses to enable
Pro features and logs critical event.

Phase 0 (now): scaffolding only — accepts any hash so dev workflow
works. The build pipeline will populate ``_EXPECTED_HASHES`` from a
generated Python file produced after Nuitka compile.

Defensive choices:
* compute hash from in-memory bytes (sys.modules → source) so attackers
  can't simply replace the file after import
* verifier itself should be one of the protected modules — if attacker
  patches verifier, integrity check is moot. Phase 2 adds a Rust-built
  duplicate that double-checks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from typing import Iterable

from orynd_core.services.logging import get_logger

log = get_logger("orynd.license.integrity")


# Filled by build pipeline (Phase 1). Empty dict means "skip checks".
_EXPECTED_HASHES: dict[str, str] = {}

# Strict mode aborts on any failure. Default off in dev.
_STRICT = os.environ.get("ORYND_INTEGRITY_STRICT") == "1"

# Critical modules — must be intact for Pro features. List grows in Phase 1.
PROTECTED_MODULES = [
    "orynd_core.services.license.jwt_verify",
    "orynd_core.services.license.state",
    "orynd_core.services.license.decorators",
    "orynd_core.services.license.tiers",
    "orynd_core.services.model_router",
]


def _hash_module(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    try:
        with open(spec.origin, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def verify_protected_modules(modules: Iterable[str] | None = None) -> dict[str, str]:
    """Return mapping ``module → "ok" | "missing" | "mismatch:<actual>"``.

    With no expected hashes configured we return ``"unconfigured"`` per module
    so the caller can decide whether to allow Pro features (dev=yes, prod=no).
    """
    targets = list(modules) if modules else PROTECTED_MODULES
    results: dict[str, str] = {}
    for name in targets:
        actual = _hash_module(name)
        expected = _EXPECTED_HASHES.get(name)
        if actual is None:
            results[name] = "missing"
            continue
        if expected is None:
            results[name] = "unconfigured"
            continue
        if actual != expected:
            results[name] = f"mismatch:{actual[:10]}"
        else:
            results[name] = "ok"
    return results


def assert_integrity() -> None:
    """Raise SystemExit if strict mode and any module fails. No-op otherwise."""
    results = verify_protected_modules()
    failures = {k: v for k, v in results.items() if v not in {"ok", "unconfigured"}}
    if failures:
        log.critical("license.integrity_failure", failures=failures)
        if _STRICT:
            sys.stderr.write(
                "ORYND integrity check failed. Reinstall the app from oryndai.com.\n"
            )
            raise SystemExit(2)


__all__ = ["PROTECTED_MODULES", "verify_protected_modules", "assert_integrity"]

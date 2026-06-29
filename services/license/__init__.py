"""License & anti-piracy package.

Public surface kept small. Imports are lazy so importing this package
doesn't pull crypto deps unless actually verifying.
"""

from orynd_core.services.license.hwid import compute_hwid
from orynd_core.services.license.state import (
    LicenseState,
    clear_license_state,
    get_license_state,
    load_license_jwt,
)
from orynd_core.services.license.tiers import (
    FEATURE_GATES,
    Tier,
    tier_includes,
)

__all__ = [
    "Tier",
    "FEATURE_GATES",
    "tier_includes",
    "LicenseState",
    "compute_hwid",
    "get_license_state",
    "load_license_jwt",
    "clear_license_state",
]

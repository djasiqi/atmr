"""Codes machine stables pour refus bootstrap (`access_denied_code`)."""

from __future__ import annotations

PENDING_ACTIVATION = "pending_activation"
DRIVER_PROFILE_INACTIVE = "driver_profile_inactive"
NO_ACTIVE_CLIENT_PROFILE = "no_active_client_profile"
INSTITUTION_INVITED = "institution_invited"
INSTITUTION_DISABLED = "institution_disabled"
DEMO_EXPIRED = "demo_expired"

ALL_CODES: frozenset[str] = frozenset(
    {
        PENDING_ACTIVATION,
        DRIVER_PROFILE_INACTIVE,
        NO_ACTIVE_CLIENT_PROFILE,
        INSTITUTION_INVITED,
        INSTITUTION_DISABLED,
        DEMO_EXPIRED,
    }
)

"""Gates d'accès commercial billing (art. 6 bis) — jamais platform_suspended."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import select

from ext import db
from models.company import Company
from models.enums import (
    PlatformBillingAccessState,
    PlatformBillingStateSource,
)
from models.platform_billing import PlatformDunningCase


class BillingCapability(str, Enum):
    RECEIVE_MARKETPLACE_OFFERS = "RECEIVE_MARKETPLACE_OFFERS"
    ACCEPT_MARKETPLACE_OFFERS = "ACCEPT_MARKETPLACE_OFFERS"
    CREATE_OWN_PORTFOLIO_BOOKING = "CREATE_OWN_PORTFOLIO_BOOKING"
    USE_BILLABLE_SUPPORT = "USE_BILLABLE_SUPPORT"
    USE_BILLABLE_CONFIGURATION = "USE_BILLABLE_CONFIGURATION"
    READ_EXISTING_BOOKING = "READ_EXISTING_BOOKING"
    UPDATE_EXISTING_BOOKING = "UPDATE_EXISTING_BOOKING"
    CANCEL_EXISTING_BOOKING = "CANCEL_EXISTING_BOOKING"
    TRACK_EXISTING_BOOKING = "TRACK_EXISTING_BOOKING"
    GPS_INGEST = "GPS_INGEST"
    DOWNLOAD_INVOICES = "DOWNLOAD_INVOICES"
    RECORD_PAYMENT = "RECORD_PAYMENT"
    EXPORT_DATA = "EXPORT_DATA"
    AUTH_SELF_SERVICE = "AUTH_SELF_SERVICE"
    DISABLE_DRIVER_OR_USER = "DISABLE_DRIVER_OR_USER"


SAFETY_CAPABILITIES_ALWAYS_ALLOWED = frozenset(
    {
        BillingCapability.READ_EXISTING_BOOKING,
        BillingCapability.UPDATE_EXISTING_BOOKING,
        BillingCapability.CANCEL_EXISTING_BOOKING,
        BillingCapability.TRACK_EXISTING_BOOKING,
        BillingCapability.GPS_INGEST,
        BillingCapability.DOWNLOAD_INVOICES,
        BillingCapability.RECORD_PAYMENT,
        BillingCapability.EXPORT_DATA,
        BillingCapability.AUTH_SELF_SERVICE,
        BillingCapability.DISABLE_DRIVER_OR_USER,
    }
)

ERROR_BILLING_ACCESS_RESTRICTED = "billing_access_restricted"


class BillingAccessRestricted(Exception):
    """Capacité commerciale bloquée par l'état d'accès billing."""

    def __init__(self, capability: BillingCapability, state: str):
        self.capability = capability
        self.state = state
        self.error_code = ERROR_BILLING_ACCESS_RESTRICTED
        super().__init__(
            f"Accès billing restreint ({state}) : {capability.value}"
        )


def _active_policy_flags(company_id: int) -> dict[str, bool]:
    """Flags partial depuis le dossier actif, sinon permissif (pas de bloc)."""
    case = db.session.scalar(
        select(PlatformDunningCase)
        .where(
            PlatformDunningCase.company_id == int(company_id),
            PlatformDunningCase.status.in_(("open", "partial", "full")),
        )
        .limit(1)
    )
    if case is None or not case.policy_snapshot:
        return {
            "partial_block_marketplace_offers": True,
            "partial_block_marketplace_acceptance": True,
            "partial_block_billable_support": True,
            "partial_block_billable_configuration": True,
        }
    snap = case.policy_snapshot
    return {
        "partial_block_marketplace_offers": bool(
            snap.get("partial_block_marketplace_offers", True)
        ),
        "partial_block_marketplace_acceptance": bool(
            snap.get("partial_block_marketplace_acceptance", True)
        ),
        "partial_block_billable_support": bool(
            snap.get("partial_block_billable_support", True)
        ),
        "partial_block_billable_configuration": bool(
            snap.get("partial_block_billable_configuration", True)
        ),
    }


def is_billing_capability_allowed(
    company_id: int,
    capability: BillingCapability,
    *,
    booking_created_at: datetime | None = None,
) -> bool:
    company = db.session.get(Company, int(company_id))
    if company is None:
        return True

    if capability in SAFETY_CAPABILITIES_ALWAYS_ALLOWED:
        # Bookings existants : cutoff state_since
        if capability in (
            BillingCapability.READ_EXISTING_BOOKING,
            BillingCapability.UPDATE_EXISTING_BOOKING,
            BillingCapability.CANCEL_EXISTING_BOOKING,
            BillingCapability.TRACK_EXISTING_BOOKING,
        ):
            state = (
                company.platform_billing_access_state
                or PlatformBillingAccessState.ACTIVE.value
            )
            if state == PlatformBillingAccessState.ACTIVE.value:
                return True
            since = company.platform_billing_state_since
            if booking_created_at is None or since is None:
                return True
            created = booking_created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            since_aware = since if since.tzinfo else since.replace(tzinfo=UTC)
            return created < since_aware
        return True

    state = (
        company.platform_billing_access_state
        or PlatformBillingAccessState.ACTIVE.value
    )
    if state == PlatformBillingAccessState.ACTIVE.value:
        return True

    flags = _active_policy_flags(company_id)

    if capability == BillingCapability.RECEIVE_MARKETPLACE_OFFERS:
        return not flags["partial_block_marketplace_offers"]
    if capability == BillingCapability.ACCEPT_MARKETPLACE_OFFERS:
        return not flags["partial_block_marketplace_acceptance"]
    if capability == BillingCapability.USE_BILLABLE_SUPPORT:
        return not flags["partial_block_billable_support"]
    if capability == BillingCapability.USE_BILLABLE_CONFIGURATION:
        return not flags["partial_block_billable_configuration"]
    if capability == BillingCapability.CREATE_OWN_PORTFOLIO_BOOKING:
        # Bloqué uniquement en full
        return state != PlatformBillingAccessState.FULL.value

    return True


def assert_billing_capability_allowed(
    company_id: int,
    capability: BillingCapability,
    *,
    booking_created_at: datetime | None = None,
) -> None:
    if not is_billing_capability_allowed(
        company_id, capability, booking_created_at=booking_created_at
    ):
        company = db.session.get(Company, int(company_id))
        state = (
            (company.platform_billing_access_state if company else None)
            or PlatformBillingAccessState.ACTIVE.value
        )
        raise BillingAccessRestricted(capability, state)


def set_billing_access_state(
    company_id: int,
    state: str,
    *,
    source: str | None,
    reason_code: str | None,
    config_id: int | None = None,
    force: bool = False,
) -> Company:
    """Transition centralisée. Priorité admin_manual > automatic_dunning."""
    company = db.session.get(Company, int(company_id))
    if company is None:
        raise ValueError("Entreprise introuvable")

    allowed_states = {m.value for m in PlatformBillingAccessState}
    if state not in allowed_states:
        raise ValueError(f"État billing invalide: {state}")

    current_source = company.platform_billing_state_source
    if (
        not force
        and source == PlatformBillingStateSource.AUTOMATIC_DUNNING.value
        and current_source == PlatformBillingStateSource.ADMIN_MANUAL.value
        and company.platform_billing_access_state
        != PlatformBillingAccessState.ACTIVE.value
    ):
        # Auto ne peut pas écraser une restriction manuelle
        return company

    if (
        not force
        and source == PlatformBillingStateSource.AUTOMATIC_DUNNING.value
        and state == PlatformBillingAccessState.ACTIVE.value
        and current_source == PlatformBillingStateSource.ADMIN_MANUAL.value
    ):
        return company

    now = datetime.now(UTC)
    company.platform_billing_access_state = state
    if state == PlatformBillingAccessState.ACTIVE.value:
        company.platform_billing_state_source = None
        company.platform_billing_state_reason_code = None
        company.platform_billing_state_since = None
        company.platform_billing_state_config_id = None
    else:
        if not source or not reason_code:
            raise ValueError("source et reason_code obligatoires si state ≠ active")
        company.platform_billing_state_source = source
        company.platform_billing_state_reason_code = reason_code
        if company.platform_billing_state_since is None:
            company.platform_billing_state_since = now
        company.platform_billing_state_config_id = config_id
    company.platform_billing_state_updated_at = now
    db.session.flush()
    return company


def is_dunning_effectively_paused(company: Company, *, now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    until = company.dunning_paused_until
    if until is None:
        return False
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    return until > now


def pause_dunning(
    company_id: int,
    *,
    until: datetime,
    reason: str,
    user_id: int | None,
) -> Company:
    company = db.session.get(Company, int(company_id))
    if company is None:
        raise ValueError("Entreprise introuvable")
    company.dunning_paused_until = until
    company.dunning_pause_reason = (reason or "")[:512]
    company.dunning_paused_by_user_id = user_id
    db.session.flush()
    return company


def clear_dunning_pause(company_id: int) -> Company:
    company = db.session.get(Company, int(company_id))
    if company is None:
        raise ValueError("Entreprise introuvable")
    company.dunning_paused_until = None
    company.dunning_pause_reason = None
    company.dunning_paused_by_user_id = None
    db.session.flush()
    return company


def billing_access_error_payload(exc: BillingAccessRestricted) -> dict[str, Any]:
    return {
        "error": str(exc),
        "error_code": ERROR_BILLING_ACCESS_RESTRICTED,
        "reason_code": ERROR_BILLING_ACCESS_RESTRICTED,
        "capability": exc.capability.value,
        "billing_access_state": exc.state,
        "retryable": False,
    }

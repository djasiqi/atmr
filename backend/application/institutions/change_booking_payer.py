"""Primitive canonique : changement de payeur institution (R07-BP-01 / R07-BP-02)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.billing.booking_payer_resolution import (
    apply_institution_payer_resolution,
    normalize_institution_target_payer,
)
from application.companies.reservations.billing_adjustment import (
    booking_billing_is_locked,
)
from application.institutions.billing_control.status import (
    reset_control_after_payer_correction,
)
from domain.billing.errors import BillingValidationError
from ext import db
from models import Booking, TransportRequest
from services.billing.billing_party_linker import is_establishment_billing_party
from services.institutions.booking_change_service import (
    _billing_snapshot,
    bump_edit_version,
    record_change_event,
)


@dataclass(frozen=True, slots=True)
class ChangeBookingPayerResult:
    ok: bool
    error: str | None = None
    status_code: int = 200
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    audit_event_id: int | None = None


def _restore_payer_fields(booking: Booking, snapshot: dict[str, Any]) -> None:
    booking.billed_to_type = snapshot.get("billed_to_type")
    booking.billed_to_company_id = snapshot.get("billed_to_company_id")
    booking.billing_party_id = snapshot.get("billing_party_id")
    booking.billing_override_reason = snapshot.get("billing_override_reason")


def assert_booking_payer_triplet_coherent(booking: Booking) -> None:
    """Invariant BP-01/BP-02 : triplet payeur atomiquement cohérent."""
    from models import BillingParty

    btype = str(getattr(booking, "billed_to_type", None) or "patient").lower().strip()
    bcomp = getattr(booking, "billed_to_company_id", None)
    bp_id = getattr(booking, "billing_party_id", None)

    if btype == "patient":
        if bcomp is not None:
            raise BillingValidationError(
                "billed_to_type=patient mais billed_to_company_id renseigné.",
                field="billed_to_company_id",
            )
        if bp_id is None:
            raise BillingValidationError(
                "billed_to_type=patient sans billing_party_id.",
                field="billing_party_id",
            )
        bp = db.session.get(BillingParty, int(bp_id))
        if bp is not None and is_establishment_billing_party(bp):
            raise BillingValidationError(
                "billed_to_type=patient avec billing_party_id établissement.",
                field="billing_party_id",
            )
        return

    if btype == "clinic":
        if not bcomp:
            raise BillingValidationError(
                "billed_to_type=clinic sans billed_to_company_id.",
                field="billed_to_company_id",
            )
        if bp_id is None:
            raise BillingValidationError(
                "billed_to_type=clinic sans billing_party_id.",
                field="billing_party_id",
            )
        bp = db.session.get(BillingParty, int(bp_id))
        if bp is not None and not is_establishment_billing_party(bp):
            raise BillingValidationError(
                "billed_to_type=clinic avec billing_party_id non-établissement.",
                field="billing_party_id",
            )
        return

    raise BillingValidationError(
        f"billed_to_type non supporté pour changement institution : {btype}.",
        field="billed_to_type",
    )


def change_booking_payer(
    booking: Booking,
    *,
    target_payer: str,
    transport_request: TransportRequest | None,
    institution_id: int,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
    override_reason: str,
    billing_change_reason_code: str,
    source: str = "institution_portal",
    financial_actor_role: str | None = None,
) -> ChangeBookingPayerResult:
    """Change le payeur — resolve → validate → persist → audit. Pas d'auto-validation."""
    try:
        target = normalize_institution_target_payer(target_payer)
    except ValueError as exc:
        return ChangeBookingPayerResult(ok=False, error=str(exc), status_code=400)

    locked, lock_msg = booking_billing_is_locked(booking)
    if locked:
        return ChangeBookingPayerResult(
            ok=False,
            error=lock_msg or "Facturation non modifiable.",
            status_code=409,
        )
    from application.invoices.booking_dispute.freeze import (
        financial_change_blocked_by_dispute,
    )

    frozen, freeze_msg = financial_change_blocked_by_dispute(booking)
    if frozen:
        return ChangeBookingPayerResult(
            ok=False,
            error=freeze_msg or "Contestation en cours : payeur gelé.",
            status_code=409,
        )

    if transport_request is None:
        resolve_fn = getattr(booking, "_resolve_source_transport_request", None)
        transport_request = resolve_fn() if callable(resolve_fn) else None
    if transport_request is None:
        return ChangeBookingPayerResult(
            ok=False,
            error="Booking non associé à une demande institution.",
            status_code=404,
        )

    before = _billing_snapshot(booking)
    control_before_status = getattr(booking, "institution_control_status", None)

    savepoint = db.session.begin_nested()
    try:
        apply_institution_payer_resolution(
            booking,
            target_billed_to_type=target,
            transport_request=transport_request,
        )
        assert_booking_payer_triplet_coherent(booking)
        reset_control_after_payer_correction(booking)
    except (BillingValidationError, ValueError) as exc:
        savepoint.rollback()
        _restore_payer_fields(booking, before)
        booking.institution_control_status = control_before_status
        return ChangeBookingPayerResult(
            ok=False,
            error=str(exc),
            status_code=422,
            before=before,
        )
    except Exception:
        savepoint.rollback()
        _restore_payer_fields(booking, before)
        booking.institution_control_status = control_before_status
        raise

    booking.billing_override_reason = override_reason.strip()
    bump_edit_version(booking)
    after = _billing_snapshot(booking)

    fin_role = financial_actor_role
    if fin_role is None and actor_role:
        fin_role = "billing" if actor_role == "institution_billing" else "admin"

    event = record_change_event(
        booking=booking,
        transport_request=transport_request,
        institution_id=institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="billing_changed",
        change_scope="billing",
        source=source,
        before_snapshot=before,
        after_snapshot=after,
        reason=override_reason.strip(),
        change_class="major",
        severity="WARNING",
        financial_actor_role=fin_role,
        billing_change_reason_code=billing_change_reason_code,
    )

    return ChangeBookingPayerResult(
        ok=True,
        before=before,
        after=after,
        status_code=200,
        audit_event_id=int(event.id) if getattr(event, "id", None) else None,
    )

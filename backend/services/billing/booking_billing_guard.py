"""Validation facturation Booking avant écriture (ex. assignation chauffeur).

Ne modifie jamais les champs de facturation : l'assignation ne décide pas du payeur.
"""

from __future__ import annotations

from typing import Any

from domain.billing.errors import BillingValidationError

_COMPANY_ID_ZERO = 0


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _as_positive_int(value: Any) -> int | None:
    if value is None or value is False:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n <= _COMPANY_ID_ZERO:
        return None
    return n


def billing_type_normalized(booking: Any) -> str:
    raw = _as_str(getattr(booking, "billed_to_type", None)) or "patient"
    return raw.strip().lower() or "patient"


def assert_non_patient_billing_complete(
    booking: Any,
    *,
    context: str,
    require_billing_party_for_clinic: bool = True,
) -> None:
    """Lève ``BillingValidationError`` si facturation non-patient incomplète."""
    btype = billing_type_normalized(booking)
    if btype == "patient":
        return

    booking_id = getattr(booking, "id", None)
    if not _as_positive_int(getattr(booking, "billed_to_company_id", None)):
        raise BillingValidationError(
            (
                f"Facturation incomplète ({context}) : billed_to_company_id "
                f"obligatoire pour billed_to_type='{btype}'"
                + (f" (booking={booking_id})" if booking_id else "")
                + "."
            ),
            field="billed_to_company_id",
        )

    if (
        require_billing_party_for_clinic
        and btype == "clinic"
        and not _as_positive_int(getattr(booking, "billing_party_id", None))
    ):
        raise BillingValidationError(
            (
                f"Facturation incomplète ({context}) : billing_party_id "
                f"obligatoire pour billed_to_type='clinic'"
                + (f" (booking={booking_id})" if booking_id else "")
                + "."
            ),
            field="billing_party_id",
        )


def validate_booking_billing_ready_for_write(booking: Any) -> None:
    """Validation assignation : aucune mutation, erreur 422 si incomplet."""
    assert_non_patient_billing_complete(
        booking,
        context="assignation chauffeur",
        require_billing_party_for_clinic=True,
    )


def ensure_booking_billing_ready_for_write(
    booking: Any,
    *,
    repair: bool = False,
) -> bool:
    """Alias rétrocompatible — validation seule (``repair`` ignoré)."""
    del repair
    validate_booking_billing_ready_for_write(booking)
    return False

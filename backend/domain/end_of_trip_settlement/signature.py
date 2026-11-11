"""Signature de facturation directe. Distincte de l'attestation de trajet."""

from __future__ import annotations

from dataclasses import dataclass

_PATIENT_ROLES = frozenset({"patient"})
_REPRESENTATIVE = "authorized_representative"


@dataclass(frozen=True, slots=True)
class BillingSignatureReview:
    accepted: bool
    reason: str
    payment_received: bool
    invoice_accepted: bool
    signature_recorded: bool
    service_cancelled: bool


def review_billing_signature(
    *,
    signatory_role: str,
    signatory_user_id: int | None,
    driver_user_id: int | None,
    text_present: bool,
    version_present: bool,
) -> BillingSignatureReview:
    """Une facture signée n'est pas un paiement reçu."""
    if signatory_role == "driver" or (
        driver_user_id is not None and signatory_user_id == driver_user_id
    ):
        return BillingSignatureReview(
            accepted=False,
            reason="driver_cannot_sign",
            payment_received=False,
            invoice_accepted=False,
            signature_recorded=False,
            service_cancelled=False,
        )
    if signatory_role == _REPRESENTATIVE:
        return BillingSignatureReview(
            accepted=False,
            reason="representative_needs_administrative_review",
            payment_received=False,
            invoice_accepted=False,
            signature_recorded=False,
            service_cancelled=False,
        )
    if signatory_role not in _PATIENT_ROLES:
        return BillingSignatureReview(
            accepted=False,
            reason="signatory_not_allowed",
            payment_received=False,
            invoice_accepted=False,
            signature_recorded=False,
            service_cancelled=False,
        )
    if not text_present or not version_present:
        return BillingSignatureReview(
            accepted=False,
            reason="signature_text_required",
            payment_received=False,
            invoice_accepted=False,
            signature_recorded=False,
            service_cancelled=False,
        )
    return BillingSignatureReview(
        accepted=True,
        reason="signature_recorded_for_review",
        payment_received=False,
        invoice_accepted=False,
        signature_recorded=True,
        service_cancelled=False,
    )


def payer_change_conflict(
    *, financial_responsibility_locked: bool, collection_in_progress: bool
) -> str | None:
    """Appelée par le garde transactionnel, pas par un écran."""
    if collection_in_progress:
        return "collection_in_progress"
    if financial_responsibility_locked:
        return "payer_locked"
    return None

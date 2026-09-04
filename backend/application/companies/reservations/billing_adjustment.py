"""Ajustement facturation (montant, billed_to) par l'entreprise de transport — PATCH dédié."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ext import db
from models import Invoice, InvoiceLine
from models.booking import Booking
from models.company import Company
from models.enums import BookingCreatedVia, InvoiceStatus

from ._status import status_value

_MIN_NONZERO_ADJUSTMENT_CHF = 0.5


def _active_invoice_line_exists(booking_id: int) -> bool:
    return (
        db.session.query(InvoiceLine)
        .join(Invoice, InvoiceLine.invoice_id == Invoice.id)
        .filter(
            InvoiceLine.reservation_id == booking_id,
            Invoice.status != InvoiceStatus.CANCELLED,
        )
        .first()
        is not None
    )


def booking_billing_is_locked(booking: Booking) -> tuple[bool, str | None]:
    if getattr(booking, "billing_locked_at", None) is not None:
        return (
            True,
            "La facturation de cette réservation est verrouillée (billing_locked_at).",
        )
    if getattr(booking, "invoice_line_id", None) is not None:
        return (
            True,
            "Cette réservation est déjà rattachée à une ligne de facture.",
        )
    if _active_invoice_line_exists(int(booking.id)):
        return (
            True,
            "Une facture non annulée contient déjà une ligne pour cette réservation.",
        )
    return False, None


def _company_may_adjust_billing_by_origin(booking: Any) -> tuple[bool, str | None]:
    """Ajustement patient/clinique : dispatch + courses institution acceptées.

    Bloqué pour invité / portail client / API partenaire (déjà payés ou hors périmètre).
    """
    raw = getattr(booking, "created_via", None)
    if raw is None:
        return True, None
    v = raw.value if isinstance(raw, BookingCreatedVia) else str(raw).lower()
    blocked = {
        BookingCreatedVia.PUBLIC_GUEST.value,
        BookingCreatedVia.CLIENT_APP.value,
        BookingCreatedVia.API_PARTNER.value,
    }
    if v in blocked:
        return (
            False,
            (
                "L'ajustement du destinataire de facture n'est disponible que pour les courses "
                "créées par l'entreprise (dispatch) ou acceptées depuis une institution. "
                "Les demandes invité, portail client ou partenaire API ne sont pas modifiables ici."
            ),
        )
    return True, None


def _billed_to_type_to_billing_intent(billed_to_type: str) -> str:
    """Mappe billed_to_type booking → billing_intent institution."""
    t = (billed_to_type or "patient").lower().strip()
    if t == "clinic":
        return "institution"
    if t == "insurance":
        return "insurance"
    return "patient"


def _apply_billing_party_resolution(
    booking: Booking,
    *,
    target_btype: str,
) -> None:
    """Ré-attache billing_party_id après changement de destinataire."""
    company_id = getattr(booking, "company_id", None)
    if company_id is None:
        return

    from services.billing.billing_party_linker import (
        ensure_patient_destination_billing_party,
        resolve_billing_party_for_clinic,
    )

    resolve_fn = getattr(booking, "_resolve_source_transport_request", None)
    transport_request = resolve_fn() if callable(resolve_fn) else None
    if transport_request is not None:
        from services.billing.institution_billing_resolver import (
            resolve_billing_party_for_institution_booking,
        )

        resolve_billing_party_for_institution_booking(
            booking=booking,
            transport_request=transport_request,
            company_id=int(company_id),
            billing_intent_override=_billed_to_type_to_billing_intent(target_btype),
        )
        # Filet : bascule patient ne doit jamais laisser un BP établissement.
        if target_btype == "patient":
            ensure_patient_destination_billing_party(booking)
        return

    if target_btype == "patient":
        ensure_patient_destination_billing_party(booking)
        return

    if target_btype == "clinic":
        clinic_id = getattr(booking, "billed_to_company_id", None)
        if clinic_id is None:
            return
        bp = resolve_billing_party_for_clinic(
            company_id=int(company_id),
            clinic_company_id=int(clinic_id),
        )
        if bp is not None:
            booking.billing_party_id = int(bp.id)


def _copy_payer_fields(source: Booking, target: Booking, *, reason: str) -> None:
    """Copie destinataire / BP / motif (pas le montant) d'un booking vers un autre."""
    target.billed_to_type = source.billed_to_type
    target.billed_to_company_id = source.billed_to_company_id
    target.billing_party_id = source.billing_party_id
    target.billing_override_reason = reason


def _propagate_payer_to_return_legs(
    outbound: Booking,
    *,
    reason: str,
    terminal_exclude: frozenset[str],
) -> list[int]:
    """Si ajustement sur l'aller : propage le payeur aux retours non verrouillés.

    Ajustement uniquement sur un retour : aucun effet sur l'aller (indépendant / différé).
    """
    if bool(getattr(outbound, "is_return", False)):
        return []
    if getattr(outbound, "id", None) is None:
        return []

    children = (
        Booking.query.filter_by(parent_booking_id=int(outbound.id))
        .order_by(Booking.id.asc())
        .all()
    )
    propagated: list[int] = []
    for child in children:
        st = status_value(getattr(child, "status", None)).lower()
        if st in terminal_exclude:
            continue
        locked, _ = booking_billing_is_locked(child)
        if locked:
            continue
        _copy_payer_fields(outbound, child, reason=reason)
        propagated.append(int(child.id))
    return propagated


@dataclass(frozen=True, slots=True)
class BookingBillingAdjustmentResult:
    ok: bool
    error: dict[str, Any] | None = None
    status_code: int | None = None
    # Pour audit côté route
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    propagated_return_ids: list[int] = field(default_factory=list)


class CompanyBookingBillingAdjustmentUseCase:
    """PATCH billing-adjustment : ne pas mélanger avec le PUT opérationnel."""

    _TERMINAL_EXCLUDE: frozenset[str] = frozenset(
        {
            "canceled",
            "cancelled",
            "no_show",
            "rejected",
        }
    )

    def execute(
        self,
        booking: Booking,
        *,
        data: dict[str, Any],
        keys_present: set[str],
    ) -> BookingBillingAdjustmentResult:
        st = status_value(getattr(booking, "status", None)).lower()
        if st in self._TERMINAL_EXCLUDE:
            return BookingBillingAdjustmentResult(
                ok=False,
                error={
                    "error": "Impossible d'ajuster la facturation d'une réservation annulée."
                },
                status_code=400,
            )

        allow_origin, origin_msg = _company_may_adjust_billing_by_origin(booking)
        if not allow_origin:
            return BookingBillingAdjustmentResult(
                ok=False,
                error={
                    "error": origin_msg or "Ajustement de facturation non autorisé."
                },
                status_code=400,
            )

        locked, lock_msg = booking_billing_is_locked(booking)
        if locked:
            return BookingBillingAdjustmentResult(
                ok=False,
                error={"error": lock_msg or "Facturation non modifiable."},
                status_code=409,
            )
        from application.invoices.booking_dispute.freeze import (
            financial_change_blocked_by_dispute,
        )

        frozen, freeze_msg = financial_change_blocked_by_dispute(booking)
        if frozen:
            return BookingBillingAdjustmentResult(
                ok=False,
                error={"error": freeze_msg or "Contestation en cours : facturation gelée."},
                status_code=409,
            )

        override = (data.get("override_reason") or "").strip()
        if not override:
            return BookingBillingAdjustmentResult(
                ok=False,
                error={"error": "Le champ override_reason est obligatoire."},
                status_code=400,
            )

        has_amount = "amount" in keys_present and data.get("amount") is not None
        has_btype = (
            "billed_to_type" in keys_present and data.get("billed_to_type") is not None
        )
        has_bcomp = "billed_to_company_id" in keys_present

        if not (has_amount or has_btype or has_bcomp):
            return BookingBillingAdjustmentResult(
                ok=False,
                error={
                    "error": (
                        "Au moins un champ amount, billed_to_type ou "
                        "billed_to_company_id est requis."
                    )
                },
                status_code=400,
            )

        raw_old_type = getattr(booking, "billed_to_type", None) or "patient"
        old_type_str = str(raw_old_type).lower().strip()
        old = {
            "amount": float(booking.amount),
            "billed_to_type": old_type_str,
            "billed_to_company_id": getattr(booking, "billed_to_company_id", None),
            "billing_party_id": getattr(booking, "billing_party_id", None),
        }

        if has_btype and data.get("billed_to_type") is not None:
            target_btype = str(data["billed_to_type"]).lower().strip()
        else:
            target_btype = old_type_str
        if target_btype not in ("patient", "clinic", "insurance"):
            return BookingBillingAdjustmentResult(
                ok=False,
                error={
                    "error": "billed_to_type invalide (patient, clinic, insurance)."
                },
                status_code=400,
            )

        target_bcomp: int | None
        if has_bcomp:
            raw = data["billed_to_company_id"]
            if raw is None:
                target_bcomp = None
            else:
                try:
                    target_bcomp = int(raw)
                except (TypeError, ValueError):
                    return BookingBillingAdjustmentResult(
                        ok=False,
                        error={"error": "billed_to_company_id invalide."},
                        status_code=400,
                    )
        else:
            target_bcomp = old["billed_to_company_id"]

        if target_btype == "patient":
            if has_bcomp and target_bcomp is not None:
                return BookingBillingAdjustmentResult(
                    ok=False,
                    error={
                        "error": "billed_to_company_id doit être absent ou null si billed_to_type vaut patient."
                    },
                    status_code=400,
                )
            target_bcomp = None
        else:
            if target_bcomp is None or (
                isinstance(target_bcomp, int) and target_bcomp <= 0
            ):
                return BookingBillingAdjustmentResult(
                    ok=False,
                    error={
                        "error": f"billed_to_company_id est obligatoire et strictement positif pour billed_to_type={target_btype}."
                    },
                    status_code=400,
                )
            c = db.session.get(Company, int(target_bcomp))
            if c is None:
                return BookingBillingAdjustmentResult(
                    ok=False,
                    error={
                        "error": "billed_to_company_id : entreprise cible introuvable."
                    },
                    status_code=400,
                )

        if has_amount:
            try:
                amt = float(data["amount"])
            except (TypeError, ValueError):
                return BookingBillingAdjustmentResult(
                    ok=False,
                    error={"error": "amount invalide."},
                    status_code=400,
                )
            if amt < 0 or (amt > 0 and amt < _MIN_NONZERO_ADJUSTMENT_CHF):
                return BookingBillingAdjustmentResult(
                    ok=False,
                    error={"error": "Le montant doit être nul ou au moins 0,50 CHF."},
                    status_code=400,
                )
            booking.amount = round(amt, 2)

        # Propagation A/R et re-résolution BP uniquement si le destinataire change réellement
        # (le formulaire envoie toujours billed_to_type même pour un simple changement de montant).
        payer_changed = (
            target_btype != old_type_str or target_bcomp != old["billed_to_company_id"]
        )

        booking.billed_to_type = target_btype
        booking.billed_to_company_id = target_bcomp
        booking.billing_override_reason = override

        if payer_changed:
            _apply_billing_party_resolution(booking, target_btype=target_btype)
            propagated_ids = _propagate_payer_to_return_legs(
                booking,
                reason=override,
                terminal_exclude=self._TERMINAL_EXCLUDE,
            )
        else:
            propagated_ids = []

        financial_changed = payer_changed or (
            has_amount and float(booking.amount) != float(old["amount"])
        )
        if financial_changed:
            from application.invoices.institution_invoice_eligibility import (
                reopen_market_lirie_validation_after_financial_change,
            )

            reopen_market_lirie_validation_after_financial_change(booking)

        after = {
            "amount": float(booking.amount),
            "billed_to_type": str(getattr(booking, "billed_to_type", None) or "patient")
            .lower()
            .strip(),
            "billed_to_company_id": getattr(booking, "billed_to_company_id", None),
            "billing_party_id": getattr(booking, "billing_party_id", None),
            "propagated_return_ids": propagated_ids,
        }

        return BookingBillingAdjustmentResult(
            ok=True,
            before=old,
            after=after,
            status_code=200,
            propagated_return_ids=propagated_ids,
        )

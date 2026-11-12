"""Contrôles communs. Le payeur et le montant viennent de la réservation."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Any

from sqlalchemy import select

from application.invoices.issued_invoice_guard import invoice_is_issued
from domain.end_of_trip_settlement.production_prep_barrier import (
    PrepRefusal,
    prep_open,
)
from ext import db
from models.enums import BookingStatus
from models.invoice import Invoice, InvoiceLine

_CENT = Decimal("0.01")
_OPERABLE = frozenset({BookingStatus.IN_PROGRESS, BookingStatus.COMPLETED})
_CANCELLED = frozenset({BookingStatus.CANCELED})


def require_prep() -> None:
    if not prep_open():
        from domain.end_of_trip_settlement.production_prep_barrier import PrepClosed

        raise PrepClosed()


def decision_version(booking: Any) -> str:
    raw = "|".join(
        (
            str(booking.id),
            str(booking.company_id),
            str(booking.price_amount),
            str(getattr(booking, "billed_to_type", None)),
            str(getattr(booking, "status", None)),
            str(bool(getattr(booking, "is_round_trip", False))),
            str(getattr(booking, "parent_booking_id", None)),
        )
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def contractual_price_label(price_amount: Any) -> str | None:
    """Libellé du prix contractuel.

    ``amount`` n'est pas une source de remplacement : un montant opérationnel
    non confirmé ne devient pas un dû. None, zéro et un montant négatif
    restent absents. La caisse refuse ces cas par ``montant_absent``, et
    l'écran ne doit pas afficher 0 CHF à la place.
    """
    if price_amount is None:
        return None
    try:
        amount = Decimal(str(price_amount)).quantize(_CENT)
    except Exception:
        return None
    if amount <= 0:
        return None
    return format(amount, "f")


def amount_due(booking: Any) -> Decimal:
    raw = getattr(booking, "price_amount", None)
    if raw is None:
        raise PrepRefusal("montant_absent", 409)
    amount = Decimal(str(raw)).quantize(_CENT)
    if amount <= 0:
        raise PrepRefusal("montant_absent", 409)
    return amount


def debtor_label(booking: Any) -> str:
    name = str(getattr(booking, "customer_name", "") or "").strip()
    return name or "Patient"


def status_of(booking: Any) -> BookingStatus | str:
    status = booking.status
    if isinstance(status, BookingStatus):
        return status
    try:
        return BookingStatus(str(status))
    except ValueError:
        return str(status)


def authorize_settlement_history_read(
    booking: Any, *, driver_id: int, company_id: int
) -> None:
    """ACL seule pour consulter un registre déjà enregistré.

    N'ouvre aucune nouvelle collecte : ``facture_emise``, ``deja_paye`` et les
    autres gardes d'écriture restent hors de ce contrôle. Une entreprise
    étrangère ou un chauffeur non affecté est toujours refusé.
    """
    if int(booking.company_id) != int(company_id):
        raise PrepRefusal("autre_entreprise", 404)
    if booking.driver_id is None or int(booking.driver_id) != int(driver_id):
        raise PrepRefusal("chauffeur_non_affecte", 403)


def refuse_booking(booking: Any, *, driver_id: int, company_id: int) -> None:
    authorize_settlement_history_read(
        booking, driver_id=driver_id, company_id=company_id
    )
    if bool(getattr(booking, "is_round_trip", False)) or bool(
        getattr(booking, "parent_booking_id", None)
    ):
        from application.settlement.leg_settlement import separate_leg_ready

        if not separate_leg_ready(booking):
            raise PrepRefusal("aller_retour_bloque", 409)
    status = status_of(booking)
    if status in _CANCELLED:
        raise PrepRefusal("transport_annule", 409)
    if status not in _OPERABLE:
        raise PrepRefusal("statut_incompatible", 409)
    if str(getattr(booking, "billed_to_type", "") or "") == "clinic":
        raise PrepRefusal("institution_facturee", 409)
    coverage = _payment_coverage(booking)
    if coverage in {"full", "surplus"}:
        raise PrepRefusal("deja_paye", 409)
    if coverage == "ambiguous":
        raise PrepRefusal("couverture_ambigue", 409)
    if _issued_invoice(int(booking.id)):
        raise PrepRefusal("facture_emise", 409)
    amount_due(booking)


def reject_client_overrides(payload: dict, *, due: Decimal, company_id: int) -> None:
    if payload.get("company_id") not in (None, company_id):
        raise PrepRefusal("entreprise_client_ignoree", 400)
    raw = payload.get("amount_chf")
    if raw is None:
        return
    try:
        offered = Decimal(str(raw)).quantize(_CENT)
    except Exception as exc:
        raise PrepRefusal("montant_client_ignore", 400) from exc
    if offered != due:
        raise PrepRefusal("montant_client_ignore", 400)


def fingerprint(payload: dict) -> str:
    kept = {
        key: payload.get(key)
        for key in (
            "action",
            "tendered_chf",
            "change_chf",
            "change_confirmed",
            "disposition",
            "counted_chf",
            "kind",
            "signatory_role",
            "reason",
        )
    }
    for extra in ("net_chf", "compensates_collection_id"):
        if payload.get(extra) is not None:
            kept[extra] = payload.get(extra)
    return hashlib.sha256(
        json.dumps(kept, sort_keys=True, default=str).encode()
    ).hexdigest()


def _payment_coverage(booking: Any) -> str:
    """Couverture Saferpay de cette course, y compris un aller-retour lié.

    Un paiement partiel ne bloque pas. Un paiement qui égale le prix, un
    excédent, ou un paiement de groupe sans périmètre clair bloque.
    Le paiement d'une autre course non liée n'entre pas dans le calcul.
    """
    from application.invoices.confirmed_payment_coverage import (
        _AMBIGUOUS,
        _FULL,
        classify_coverage,
        confirmed_amounts_by_booking,
        group_override,
        linked_round_trip,
    )

    booking_id = int(booking.id)
    legs = linked_round_trip(booking)
    ids = [int(leg.id) for leg in legs] if legs else [booking_id]
    covered = confirmed_amounts_by_booking(ids)
    if legs:
        override = group_override(legs, covered)
        if override in {_FULL, _AMBIGUOUS}:
            return override
    raw = getattr(booking, "price_amount", None)
    due = Decimal(str(raw)).quantize(_CENT) if raw is not None else None
    if due is not None and due <= 0:
        due = None
    paid = covered.get(booking_id, Decimal("0.00"))
    return classify_coverage(due, paid)


def require_new_cash_declaration(
    booking: Any, *, driver_id: int, company_id: int
) -> None:
    """Refuse une nouvelle déclaration si la lecture ne propose pas les espèces.

    À appeler seulement quand aucune ligne n'est encore rejouée. Un dossier
    déjà enregistré reste lisible et rejouable sans ce contrôle.
    """
    from application.settlement.driver_read_contract import (
        cash_mode_offered,
        driver_settlement_contract,
    )

    class _Reader:
        def __init__(self) -> None:
            self.id = int(driver_id)
            self.company_id = int(company_id)

    contract = driver_settlement_contract(int(booking.id), _Reader())
    if cash_mode_offered(contract):
        return
    from domain.end_of_trip_settlement.v1_local_ledger import v1_local_ledger_enabled

    if v1_local_ledger_enabled():
        refuse_booking(booking, driver_id=int(driver_id), company_id=int(company_id))
        sheet = (contract or {}).get("cash_sheet")
        if (
            isinstance(sheet, dict)
            and sheet.get("allowed") is True
            and sheet.get("register") == "cash_v1"
            and sheet.get("simulation") is not True
            and sheet.get("real_collection") is not True
        ):
            return
    raise PrepRefusal("especes_non_autorisees", 409)


def _issued_invoice(booking_id: int) -> bool:
    invoices = db.session.execute(
        select(Invoice)
        .join(InvoiceLine, InvoiceLine.invoice_id == Invoice.id)
        .where(InvoiceLine.reservation_id == booking_id)
    ).scalars()
    return any(invoice_is_issued(invoice) for invoice in invoices)

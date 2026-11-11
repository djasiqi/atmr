"""Couverture d'une prestation par des paiements confirmés.

Un paiement COMPLETED n'est compté que s'il est lié à cette réservation.
Le montant couvert est comparé au prix facturable de cette réservation.
Une couverture partielle ou supérieure au prix est signalée : elle n'est
pas répartie sur d'autres courses, ni soldée par un montant inventé.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, func, or_

from application.invoices.billable_amount import calculate_billable_booking_amount
from ext import db
from models.enums import PaymentStatus

_CENT = Decimal("0.01")
_FULL = "full"
_SURPLUS = "surplus"
_PARTIAL = "partial"
_AMBIGUOUS = "ambiguous"
_NONE = "none"


def _money(value: Any) -> Decimal:
    return Decimal(str(value or 0)).quantize(_CENT)


def _captured_amount(payment: Any) -> Decimal:
    captured = getattr(payment, "saferpay_captured_amount", None)
    if captured is not None:
        return _money(captured)
    return _money(getattr(payment, "amount", None))


def confirmed_amounts_by_booking(booking_ids: list[int]) -> dict[int, Decimal]:
    """Somme des paiements COMPLETED de chaque réservation, rien d'autre."""
    if not booking_ids:
        return {}
    from models.payment import Payment

    rows = Payment.query.filter(
        Payment.booking_id.in_(booking_ids),
        Payment.status == PaymentStatus.COMPLETED,
    ).all()
    totals: dict[int, Decimal] = defaultdict(lambda: Decimal("0.00"))
    for row in rows:
        booking_id = getattr(row, "booking_id", None)
        if booking_id is None:
            continue
        totals[int(booking_id)] += _captured_amount(row)
    return {key: value.quantize(_CENT) for key, value in totals.items()}


def classify_coverage(due: Decimal | None, covered: Decimal) -> str:
    """none, partial, full, surplus ou ambiguous."""
    paid = _money(covered)
    if paid <= 0:
        return _NONE
    if due is None or _money(due) <= 0:
        return _AMBIGUOUS
    price = _money(due)
    if paid == price:
        return _FULL
    if paid > price:
        return _SURPLUS
    return _PARTIAL


def _due_amount(booking: Any, billing_settings: Any) -> Decimal | None:
    billed = calculate_billable_booking_amount(
        booking, billing_settings=billing_settings
    )
    if not billed.resolved:
        return None
    return billed.amount_ht


def _same_company(left: Any, right: Any) -> bool:
    return getattr(left, "company_id", None) == getattr(right, "company_id", None)


def _round_trip_legs(bookings: list[Any]) -> dict[int, list[Any]]:
    """Trajets liés par parent_booking_id, dans la même entreprise.

    Un paiement n'est pas ventilé vers une réservation sans ce lien.
    """
    from models.booking import Booking

    seeds: set[int] = set()
    for booking in bookings:
        booking_id = getattr(booking, "id", None)
        if booking_id is None:
            continue
        seeds.add(int(booking_id))
        parent_id = getattr(booking, "parent_booking_id", None)
        if parent_id is not None:
            seeds.add(int(parent_id))
    if not seeds:
        return {}
    children = Booking.query.filter(Booking.parent_booking_id.in_(seeds)).all()
    parent_ids = {int(child.parent_booking_id) for child in children if child.parent_booking_id}
    if not parent_ids:
        return {}
    parents = {
        int(parent.id): parent
        for parent in Booking.query.filter(Booking.id.in_(parent_ids)).all()
    }
    grouped: dict[int, list[Any]] = {}
    for parent_id, parent in parents.items():
        legs = [parent]
        for child in children:
            if int(child.parent_booking_id) != parent_id:
                continue
            if not _same_company(parent, child):
                continue
            legs.append(child)
        if len(legs) < 2:
            continue
        for leg in legs:
            grouped[int(leg.id)] = legs
    return grouped


def _group_override(
    legs: list[Any],
    covered: dict[int, Decimal],
    billing_settings: Any,
) -> str | None:
    """full si le paiement égale la somme du groupe, ambiguous s'il la dépasse
    sans l'égaler, None si chaque paiement reste dans le prix de sa course.
    """
    dues: dict[int, Decimal | None] = {
        int(leg.id): _due_amount(leg, billing_settings) for leg in legs
    }
    if any(due is None or _money(due) <= 0 for due in dues.values()):
        if any(covered.get(int(leg.id), Decimal("0.00")) > 0 for leg in legs):
            return _AMBIGUOUS
        return None
    overpaid = False
    group_paid = Decimal("0.00")
    group_due = Decimal("0.00")
    for leg in legs:
        leg_id = int(leg.id)
        paid = covered.get(leg_id, Decimal("0.00"))
        due = _money(dues[leg_id])
        group_paid += paid
        group_due += due
        if paid > due:
            overpaid = True
    if not overpaid:
        return None
    if _money(group_paid) == _money(group_due):
        return _FULL
    return _AMBIGUOUS


def linked_round_trip(booking: Any) -> list[Any] | None:
    """Trajets liés à cette réservation, ou None s'il n'y a pas de groupe."""
    booking_id = getattr(booking, "id", None)
    if booking_id is None:
        return None
    return _round_trip_legs([booking]).get(int(booking_id))


def group_override(legs: list[Any], covered: dict[int, Decimal]) -> str | None:
    """Périmètre d'un paiement groupé. None si chaque paiement reste dans son prix."""
    return _group_override(legs, covered, None)


def apply_confirmed_payment_coverage(
    bookings: list[Any],
    warnings: list[str] | None = None,
    *,
    billing_settings: Any = None,
) -> list[Any]:
    """Retire les prestations intégralement couvertes et signale le reste.

    Une couverture partielle reste sélectionnable. Un excédent sur une course
    sans retour lié retire cette course sans créditer une autre réservation.
    Un paiement qui égale la somme d'un aller-retour lié retire les deux
    trajets. Un écart sur ce groupe suspend la facturation, sans ventilation.
    """
    notes = warnings if warnings is not None else []
    if not bookings:
        return []
    ids = [int(booking.id) for booking in bookings if getattr(booking, "id", None)]
    announced_groups: set[int] = set()
    groups = _round_trip_legs(bookings)
    group_ids = {
        int(leg.id) for legs in groups.values() for leg in legs
    }
    covered = confirmed_amounts_by_booking(ids + list(group_ids))
    kept: list[Any] = []
    for booking in bookings:
        booking_id = getattr(booking, "id", None)
        if booking_id is None:
            kept.append(booking)
            continue
        legs = groups.get(int(booking_id))
        if legs:
            override = _group_override(legs, covered, billing_settings)
            if override == _FULL:
                anchor = int(legs[0].id)
                if anchor not in announced_groups:
                    announced_groups.add(anchor)
                    paid = sum(
                        (covered.get(int(leg.id), Decimal("0.00")) for leg in legs),
                        Decimal("0.00"),
                    )
                    labels = ", ".join(f"#{int(leg.id)}" for leg in legs)
                    notes.append(
                        f"Aller-retour {labels} couvert par un paiement confirmé "
                        f"de {_money(paid)} CHF : trajets retirés de la facturation."
                    )
                continue
            if override == _AMBIGUOUS:
                notes.append(
                    f"Transport #{int(booking_id)} : paiement d'aller-retour "
                    "sans périmètre égal à la somme des trajets. Facturation "
                    "suspendue, aucune ventilation automatique."
                )
                continue
        paid = covered.get(int(booking_id), Decimal("0.00"))
        kind = classify_coverage(_due_amount(booking, billing_settings), paid)
        if kind == _NONE:
            kept.append(booking)
            continue
        if kind == _FULL:
            notes.append(
                f"Transport #{int(booking_id)} déjà réglé ({paid} CHF) : "
                "retiré de la facturation."
            )
            continue
        if kind == _SURPLUS:
            notes.append(
                f"Transport #{int(booking_id)} : paiement confirmé de {paid} CHF "
                "supérieur au prix de cette course. Prestation non proposée. "
                "L'excédent n'est pas affecté à une autre course."
            )
            continue
        if kind == _PARTIAL:
            notes.append(
                f"Transport #{int(booking_id)} : paiement confirmé de {paid} CHF, "
                "couverture partielle. La prestation reste proposée, "
                "le solde n'est pas soldé automatiquement."
            )
            kept.append(booking)
            continue
        notes.append(
            f"Transport #{int(booking_id)} : couverture de paiement ambiguë. "
            "La prestation reste proposée."
        )
        kept.append(booking)
    return kept


def fully_covered_booking_ids_query() -> Any:
    """Sous-requête des courses dont les paiements COMPLETED couvrent le prix.

    Le prix comparé est ``Booking.amount`` et, s'il est renseigné,
    ``Booking.price_amount``. Un aller-retour lié est retiré lorsque le
    paiement du parent et du retour couvre la somme des deux prix.
    Un paiement d'une autre entreprise n'entre pas dans la somme.
    """
    from sqlalchemy.orm import aliased

    from models.booking import Booking
    from models.payment import Payment

    paid = func.coalesce(func.sum(Payment.amount), 0)
    single = (
        db.session.query(Payment.booking_id.label("booking_id"))
        .join(Booking, Booking.id == Payment.booking_id)
        .filter(Payment.status == PaymentStatus.COMPLETED)
        .group_by(Payment.booking_id, Booking.amount, Booking.price_amount)
        .having(
            and_(
                paid >= Booking.amount - 0.004,
                or_(
                    Booking.price_amount.is_(None),
                    paid >= Booking.price_amount - 0.004,
                ),
            )
        )
    )
    parent = aliased(Booking)
    child = aliased(Booking)
    group_paid = (
        db.session.query(func.coalesce(func.sum(Payment.amount), 0))
        .filter(
            Payment.status == PaymentStatus.COMPLETED,
            or_(Payment.booking_id == parent.id, Payment.booking_id == child.id),
        )
        .correlate(parent, child)
        .scalar_subquery()
    )
    grouped_children = (
        db.session.query(child.id.label("booking_id"))
        .join(parent, parent.id == child.parent_booking_id)
        .filter(child.company_id.is_not_distinct_from(parent.company_id))
        .filter(group_paid >= parent.amount + child.amount - 0.004)
    )
    return single.union(grouped_children)

"""Verrou de responsabilité financière pour un règlement sur place.

Distinct du verrou de facturation (`billing_locked_at`, ligne de facture)
et du contrôle institutionnel « validé ». Aucune réservation ancienne n'est
verrouillée : sans politique active, le changement de payeur actuel continue.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import text

from domain.end_of_trip_settlement.signature import payer_change_conflict
from ext import db
from models.booking import Booking
from models.settlement_policy import BookingSettlementPolicy

ORDINARY_PAYER_CHANGE_REFUSED = (
    "Le payeur est verrouillé pour le règlement prévu. "
    "Un changement ordinaire est refusé."
)


def ordinary_payer_change_message(reason: str) -> str:
    if reason == "collection_in_progress":
        return (
            "Un encaissement est déjà en préparation. "
            "Le payeur ne peut pas changer pendant cette opération."
        )
    return ORDINARY_PAYER_CHANGE_REFUSED


def settlement_policy_table_ready() -> bool:
    """Vrai seulement si la migration de politique est présente."""
    name = db.session.execute(
        text("SELECT to_regclass('public.booking_settlement_policy')")
    ).scalar()
    return name is not None


def refuse_ordinary_payer_changes(booking_ids: tuple[int, ...]) -> str | None:
    """Refuse un changement ordinaire si une politique active est verrouillée.

    Verrouille les réservations puis les politiques actives, dans l'ordre des
    identifiants, dans la transaction courante. Sans table ou sans ligne,
    retourne None.
    """
    if not settlement_policy_table_ready():
        return None
    ids = tuple(sorted({int(item) for item in booking_ids if int(item) > 0}))
    if not ids:
        return None
    (
        db.session.query(Booking)
        .filter(Booking.id.in_(ids))
        .order_by(Booking.id.asc())
        .with_for_update()
        .all()
    )
    rows = (
        db.session.query(BookingSettlementPolicy)
        .filter(
            BookingSettlementPolicy.booking_id.in_(ids),
            BookingSettlementPolicy.status == "active",
        )
        .order_by(BookingSettlementPolicy.booking_id.asc())
        .with_for_update()
        .all()
    )
    for row in rows:
        reason = payer_change_conflict(
            financial_responsibility_locked=bool(row.financial_responsibility_locked),
            collection_in_progress=False,
        )
        if reason is not None:
            return reason
    return None


def refuse_ordinary_payer_change(booking_id: int) -> str | None:
    return refuse_ordinary_payer_changes((int(booking_id),))


def lock_financial_responsibility(*, booking_id: int, actor_user_id: int) -> str:
    """Pose un verrou explicite. Aucun portail ne l'appelle."""
    if actor_user_id < 1:
        return "actor_required"
    if not settlement_policy_table_ready():
        return "policy_table_absent"
    booking = (
        db.session.query(Booking)
        .filter(Booking.id == int(booking_id))
        .with_for_update()
        .one_or_none()
    )
    if booking is None:
        return "booking_missing"
    row = (
        db.session.query(BookingSettlementPolicy)
        .filter(
            BookingSettlementPolicy.booking_id == int(booking_id),
            BookingSettlementPolicy.status == "active",
        )
        .with_for_update()
        .one_or_none()
    )
    if row is None:
        return "settlement_policy_missing"
    if row.financial_responsibility_locked:
        return "already_locked"
    from application.companies.reservations.billing_adjustment import (
        booking_billing_is_locked,
    )

    covered, _message = booking_billing_is_locked(booking)
    if covered:
        return "invoice_or_billing_lock"
    row.financial_responsibility_locked = True
    row.locked_at = datetime.now(UTC)
    row.locked_by_user_id = int(actor_user_id)
    return "locked"

"""Visibilité réservations entreprise — chemins disjoints (owner / exécutant / transfert / offre).

Sémantique alignée sur ``BookingRepository._company_visibility_filter``.
Le ``OR`` unique force un seq scan + JIT ; l'union de branches disjointes
permet l'index ``(company_id, scheduled_time)`` sur le cas propriétaire.
Les exclusions d'exécutant sont NULL-safe (``IS DISTINCT FROM``) : un
``NOT (executing_company_id = X)`` SQL exclurait les offres ouvertes.
"""

from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy import case, select, union_all
from sqlalchemy.sql import ColumnElement

from ext import db
from models import Booking
from models.booking_transfer import BookingTransfer
from models.enums import BookingStatus, DispatchOfferStatus, TransferStatus
from models.service_area_pricing import DispatchOffer

EXECUTOR_VISIBLE_STATUSES = (
    BookingStatus.PENDING,
    BookingStatus.ACCEPTED,
    BookingStatus.ASSIGNED,
    BookingStatus.EN_ROUTE,
    BookingStatus.IN_PROGRESS,
    BookingStatus.COMPLETED,
    BookingStatus.RETURN_COMPLETED,
)

_TRANSFER_OWNER_STATUSES = (TransferStatus.ACCEPTED, TransferStatus.COMPLETED)
_OPEN_OFFER_STATUSES = (BookingStatus.PENDING, BookingStatus.AWAITING_CLIENT_PAYMENT)


def _executor_overlap(company_id: int) -> ColumnElement[bool]:
    return (Booking.executing_company_id == company_id) & Booking.status.in_(
        EXECUTOR_VISIBLE_STATUSES
    )


def _not_executor(company_id: int) -> ColumnElement[bool]:
    """NULL-safe : ``NOT (executing = id AND status visible)`` reste vrai si exécutant vide."""
    return Booking.executing_company_id.is_distinct_from(company_id) | (
        ~Booking.status.in_(EXECUTOR_VISIBLE_STATUSES)
    )


def _transfer_owner_exists(company_id: int):
    return (
        select(BookingTransfer.id)
        .where(
            BookingTransfer.booking_id == Booking.id,
            BookingTransfer.owner_company_id == company_id,
            BookingTransfer.status.in_(_TRANSFER_OWNER_STATUSES),
        )
        .exists()
    )


def _open_offer_exists(company_id: int):
    return (
        select(DispatchOffer.id)
        .where(
            DispatchOffer.booking_id == Booking.id,
            DispatchOffer.company_id == company_id,
            DispatchOffer.status == DispatchOfferStatus.PROPOSED,
        )
        .exists()
    )


def visibility_branch_predicates(company_id: int) -> dict[str, ColumnElement[bool]]:
    """Quatre prédicats disjoints dont l'union = le filtre OR historique."""
    executor_overlap = _executor_overlap(company_id)
    transfer_exists = _transfer_owner_exists(company_id)
    return {
        "owned": Booking.company_id == company_id,
        "executor": executor_overlap & Booking.company_id.is_distinct_from(company_id),
        "transfer_owner": (
            Booking.company_id.is_distinct_from(company_id)
            & _not_executor(company_id)
            & transfer_exists
        ),
        "open_offer": (
            Booking.company_id.is_(None)
            & Booking.status.in_(_OPEN_OFFER_STATUSES)
            & _not_executor(company_id)
            & ~transfer_exists
            & _open_offer_exists(company_id)
        ),
    }


def _apply_extra(where: Sequence[Any], extra_filters: Sequence[Any] | None) -> list[Any]:
    clauses = list(where)
    if extra_filters:
        clauses.extend(extra_filters)
    return clauses


def _order_columns(sort_desc: bool):
    null_rank = case((Booking.scheduled_time.is_(None), 1), else_=0)
    scheduled = (
        Booking.scheduled_time.desc() if sort_desc else Booking.scheduled_time.asc()
    )
    booking_id = Booking.id.desc() if sort_desc else Booking.id.asc()
    return null_rank, scheduled.nullslast(), booking_id


def visible_booking_id_union(
    company_id: int,
    extra_filters: Sequence[Any] | None = None,
):
    """SELECT id UNION ALL des 4 branches (pour IN / COUNT / stats)."""
    parts = []
    for predicate in visibility_branch_predicates(company_id).values():
        parts.append(
            select(Booking.id).where(*_apply_extra((predicate,), extra_filters))
        )
    return union_all(*parts)


def visible_bookings_query(
    company_id: int,
    extra_filters: Sequence[Any] | None = None,
):
    """Query Booking visible via ``id IN (union)`` — même sémantique que l'OR."""
    return Booking.query.filter(
        Booking.id.in_(visible_booking_id_union(company_id, extra_filters))
    )


def visible_booking_ids_topk(
    company_id: int,
    *,
    extra_filters: Sequence[Any] | None = None,
    sort_desc: bool = True,
    offset: int = 0,
    limit: int = 25,
) -> list[int]:
    """Page d'IDs : LIMIT poussé dans chaque branche, puis fusion."""
    window = max(int(offset) + int(limit), int(limit))
    null_rank, scheduled_ord, id_ord = _order_columns(sort_desc)
    parts = []
    for predicate in visibility_branch_predicates(company_id).values():
        parts.append(
            select(Booking.id, Booking.scheduled_time)
            .where(*_apply_extra((predicate,), extra_filters))
            .order_by(null_rank, scheduled_ord, id_ord)
            .limit(window)
        )
    merged = union_all(*parts).subquery("vis_topk")
    merged_null = case((merged.c.scheduled_time.is_(None), 1), else_=0)
    merged_sched = (
        merged.c.scheduled_time.desc() if sort_desc else merged.c.scheduled_time.asc()
    )
    merged_id = merged.c.id.desc() if sort_desc else merged.c.id.asc()
    stmt = (
        select(merged.c.id)
        .order_by(merged_null, merged_sched.nullslast(), merged_id)
        .offset(max(int(offset), 0))
        .limit(max(int(limit), 1))
    )
    return [int(row[0]) for row in db.session.execute(stmt)]

"""Préchargement batch des transferts actifs pour éviter N+1 sur listes réservations."""

from __future__ import annotations

from typing import Any

from models.booking import Booking
from models.booking_transfer import BookingTransfer
from models.enums import TransferStatus


def _transfer_info_dict(transfer: BookingTransfer) -> dict[str, Any]:
    try:
        return transfer.to_dict()
    except Exception:
        return {
            "id": getattr(transfer, "id", None),
            "status": str(getattr(transfer, "status", "")),
        }


def build_transfer_cache_for_bookings(
    bookings: list[Booking],
) -> dict[int, dict[str, Any]]:
    """Retourne booking_id → { is_transferred, active_transfer }."""
    if not bookings:
        return {}
    booking_ids = [int(b.id) for b in bookings if getattr(b, "id", None) is not None]
    if not booking_ids:
        return {}

    rows = (
        BookingTransfer.query.filter(BookingTransfer.booking_id.in_(booking_ids))
        .filter(
            BookingTransfer.status.in_(
                [
                    TransferStatus.PENDING,
                    TransferStatus.ACCEPTED,
                    TransferStatus.COMPLETED,
                ]
            )
        )
        .all()
    )

    by_booking: dict[int, list[BookingTransfer]] = {}
    for row in rows:
        bid = int(row.booking_id)
        by_booking.setdefault(bid, []).append(row)

    cache: dict[int, dict[str, Any]] = {}
    for booking in bookings:
        bid = int(booking.id)
        company_id = getattr(booking, "company_id", None)
        transfers = by_booking.get(bid, [])
        active = None
        is_transferred = False
        for t in transfers:
            status = getattr(t, "status", None)
            if status in (
                TransferStatus.PENDING,
                TransferStatus.ACCEPTED,
                TransferStatus.COMPLETED,
            ):
                if active is None:
                    active = _transfer_info_dict(t)
            if status in (TransferStatus.ACCEPTED, TransferStatus.COMPLETED):
                owner_id = getattr(t, "owner_company_id", None)
                if (
                    owner_id is not None
                    and company_id is not None
                    and owner_id != company_id
                ):
                    is_transferred = True
        if not is_transferred and getattr(booking, "executing_company_id", None):
            exec_id = booking.executing_company_id
            if company_id is not None and exec_id != company_id:
                is_transferred = True
        cache[bid] = {
            "is_transferred": is_transferred,
            "active_transfer": active,
        }
    return cache


def attach_transfer_cache_to_bookings(bookings: list[Booking]) -> None:
    cache = build_transfer_cache_for_bookings(bookings)
    for booking in bookings:
        bid = int(booking.id)
        booking._transfer_cache = cache.get(  # noqa: SLF001
            bid,
            {"is_transferred": False, "active_transfer": None},
        )


def attach_route_group_leg_counts_to_bookings(bookings: list[Booking]) -> None:
    """Précharge le nombre de legs par route_group_id (badges multi-étapes)."""
    if not bookings:
        return
    group_ids = {
        str(b.route_group_id)
        for b in bookings
        if getattr(b, "route_group_id", None)
    }
    if not group_ids:
        return
    from sqlalchemy import func

    from ext import db

    rows = (
        db.session.query(Booking.route_group_id, func.count(Booking.id))
        .filter(Booking.route_group_id.in_(group_ids))
        .group_by(Booking.route_group_id)
        .all()
    )
    counts = {str(gid): int(count) for gid, count in rows}
    for booking in bookings:
        gid = getattr(booking, "route_group_id", None)
        if gid:
            booking._route_group_leg_count = counts.get(str(gid), 1)  # noqa: SLF001


def attach_serialize_context_to_bookings(
    bookings: list[Booking],
    viewer_company_id: int | None,
) -> None:
    for booking in bookings:
        booking._serialize_viewer_company_id = viewer_company_id  # noqa: SLF001

"""Préchargement batch des transferts actifs pour éviter N+1 sur listes réservations."""

from __future__ import annotations

import logging
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
            if (
                status
                in (
                    TransferStatus.PENDING,
                    TransferStatus.ACCEPTED,
                    TransferStatus.COMPLETED,
                )
                and active is None
            ):
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
        booking._transfer_cache = cache.get(
            bid,
            {"is_transferred": False, "active_transfer": None},
        )


def _blank_medical(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "non spécifié", "non specifie", "none", "null"}:
        return ""
    return text


def _route_leg_summary(booking: Booking) -> dict[str, Any]:
    from services.companies.booking_display import build_booking_scheduling

    scheduling = build_booking_scheduling(booking)
    return {
        "id": getattr(booking, "id", None),
        "sequence": getattr(booking, "route_sequence_number", None),
        "is_return": bool(getattr(booking, "is_return", False)),
        "pickup_location": getattr(booking, "pickup_location", None),
        "dropoff_location": getattr(booking, "dropoff_location", None),
        "time_confirmed": scheduling.get("time_confirmed"),
        "time_scheduled": scheduling.get("time_scheduled"),
        "display_time": scheduling.get("display_time"),
        "appointment_time": scheduling.get("appointment_time"),
        "hospital_service": _blank_medical(getattr(booking, "hospital_service", None))
        or None,
        "doctor_name": _blank_medical(getattr(booking, "doctor_name", None)) or None,
    }


def attach_route_group_legs_to_bookings(bookings: list[Booking]) -> None:
    """Résumé de chaque étape du parcours, même si la liste n'en montre qu'une."""
    if not bookings:
        return
    group_ids = {
        str(booking.route_group_id)
        for booking in bookings
        if getattr(booking, "route_group_id", None)
    }
    if not group_ids:
        return
    rows = (
        Booking.query.filter(Booking.route_group_id.in_(list(group_ids)))
        .order_by(Booking.route_sequence_number.asc(), Booking.id.asc())
        .all()
    )
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        gid = str(row.route_group_id)
        by_group.setdefault(gid, []).append(_route_leg_summary(row))
    for booking in bookings:
        gid = getattr(booking, "route_group_id", None)
        if gid:
            booking._route_group_legs = by_group.get(str(gid), [])


def attach_route_group_leg_counts_to_bookings(bookings: list[Booking]) -> None:
    """Précharge le nombre de legs par route_group_id (badges multi-étapes)."""
    if not bookings:
        return
    group_ids = {
        str(b.route_group_id) for b in bookings if getattr(b, "route_group_id", None)
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
            booking._route_group_leg_count = counts.get(str(gid), 1)


def attach_return_leg_topology_to_bookings(bookings: list[Booking]) -> None:
    """Précharge is_return_stop par booking_id (badges retour institution)."""
    if not bookings:
        return
    booking_ids = [int(b.id) for b in bookings if getattr(b, "id", None) is not None]
    if not booking_ids:
        return

    from models.transport_request_leg import TransportRequestLeg

    legs = TransportRequestLeg.query.filter(
        TransportRequestLeg.booking_id.in_(booking_ids)
    ).all()

    topology_by_booking: dict[int, bool] = {}
    for leg in legs:
        bid = getattr(leg, "booking_id", None)
        if bid is None:
            continue
        bid_int = int(bid)
        if bool(getattr(leg, "is_return_stop", False)):
            topology_by_booking[bid_int] = True
        elif bid_int not in topology_by_booking:
            topology_by_booking[bid_int] = False

    for booking in bookings:
        bid = getattr(booking, "id", None)
        if bid is None:
            continue
        if int(bid) in topology_by_booking:
            booking._is_return_leg_from_topology = topology_by_booking[int(bid)]


def _is_open_portal_client_booking(booking: Any) -> bool:
    """Demande d'un client portail, pas encore prise par une entreprise."""
    client = getattr(booking, "client", None)
    if client is None:
        return False
    client_type = getattr(client, "client_type", None)
    value = getattr(client_type, "value", client_type)
    return str(value or "").strip().upper() == "PORTAL"


def attach_serialize_context_to_bookings(
    bookings: list[Booking],
    viewer_company_id: int | None,
) -> None:
    attach_return_leg_topology_to_bookings(bookings)
    for booking in bookings:
        booking._serialize_viewer_company_id = viewer_company_id

    if viewer_company_id is None:
        return

    from services.legal.portal_double_validation import (
        booking_uses_portal_contract_flow,
    )
    from services.pricing.portal_carrier_ceiling import (
        estimate_portal_carrier_offer_amount,
    )

    for booking in bookings:
        # Marché ouvert : tarif de la grille du viewer, jamais booking.amount
        # (estimation client) ni le plafond. Vaut pour le flux contractuel et
        # pour une demande PORTAL encore non assignée.
        if getattr(booking, "company_id", None) is not None:
            continue
        if not (
            booking_uses_portal_contract_flow(booking)
            or _is_open_portal_client_booking(booking)
        ):
            continue
        try:
            suggested = estimate_portal_carrier_offer_amount(
                booking, int(viewer_company_id)
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "attach carrier_quote failed booking_id=%s company_id=%s: %s",
                getattr(booking, "id", None),
                viewer_company_id,
                exc,
                exc_info=True,
            )
            suggested = None
        booking._company_suggested_amount = suggested
        logging.getLogger(__name__).debug(
            "attach carrier_quote booking_id=%s viewer=%s suggested=%s flow=%s",
            getattr(booking, "id", None),
            viewer_company_id,
            suggested,
            getattr(booking, "portal_contract_flow", None),
        )

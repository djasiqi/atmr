"""Projection institution : la réservation convertie fait foi à l'affichage.

La ligne ``transport_requests`` conserve la demande d'origine. Après accord,
les écrans lisent les valeurs opérationnelles de la réservation, sans réécrire
cet historique.
"""

from __future__ import annotations

from typing import Any


def portal_appointment_iso(booking: Any | None) -> str | None:
    """Heure de rendez-vous portée par la réservation quand aucun segment n'existe."""
    if booking is None:
        return None
    breakdown = getattr(booking, "price_breakdown_json", None)
    if not isinstance(breakdown, dict):
        return None
    raw = breakdown.get("portal_appointment_time")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def project_operational_reservation(
    payload: dict[str, Any],
    transport_request: Any,
    bookings_by_id: dict[int, Any] | None = None,
) -> dict[str, Any]:
    """Superpose la réservation actuelle sur la demande, pour la lecture seule."""
    out = dict(payload)
    booking = getattr(transport_request, "booking", None)
    if booking is not None:
        _overlay_booking_fields(out, booking)
        summary = dict(out.get("booking_summary") or {})
        appointment = portal_appointment_iso(booking)
        if appointment:
            out["appointment_time"] = appointment
            summary["appointment_time"] = appointment
        if summary:
            out["booking_summary"] = summary

    legs_payload = out.get("legs")
    legs = list(getattr(transport_request, "legs", None) or [])
    if isinstance(legs_payload, list) and legs:
        lookup = bookings_by_id
        if lookup is None:
            lookup = _load_leg_bookings(legs)
        projected: list[Any] = []
        for index, item in enumerate(legs_payload):
            if not isinstance(item, dict) or index >= len(legs):
                projected.append(item)
                continue
            leg_booking_id = getattr(legs[index], "booking_id", None)
            row = lookup.get(int(leg_booking_id)) if leg_booking_id else None
            projected.append(_overlay_leg_fields(dict(item), row))
        out["legs"] = projected
    return out


def _overlay_booking_fields(payload: dict[str, Any], booking: Any) -> None:
    for field in ("pickup_location", "dropoff_location"):
        value = getattr(booking, field, None)
        if isinstance(value, str) and value.strip():
            payload[field] = value
    notes = getattr(booking, "pickup_access_notes", None)
    if isinstance(notes, str) and notes.strip():
        payload["notes"] = notes.strip()
    scheduled = getattr(booking, "scheduled_time", None)
    if scheduled is not None:
        from models.transport_request import _iso_scheduled

        payload["scheduled_time"] = _iso_scheduled(scheduled)


def _overlay_leg_fields(item: dict[str, Any], booking: Any | None) -> dict[str, Any]:
    """Le segment affiche le lieu opérationnel. Son horaire reste celui du segment."""
    if booking is None:
        return item
    for field in ("pickup_location", "dropoff_location"):
        value = getattr(booking, field, None)
        if isinstance(value, str) and value.strip():
            item[field] = value
    return item


def _load_leg_bookings(legs: list[Any]) -> dict[int, Any]:
    identifiers = [
        int(booking_id)
        for booking_id in (getattr(leg, "booking_id", None) for leg in legs)
        if booking_id
    ]
    if not identifiers:
        return {}
    from models import Booking

    rows = Booking.query.filter(Booking.id.in_(identifiers)).all()
    return {int(row.id): row for row in rows}

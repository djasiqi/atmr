"""Présentation API contrôle facturation institution — aucune logique métier nouvelle."""

from __future__ import annotations

from typing import Any

from application.companies.reservations.billing_adjustment import (
    booking_billing_is_locked,
)
from application.institutions.billing_control.status import (
    control_status_snapshot,
    effective_control_status,
)
from application.invoices.booking_dispute.service import latest_dispute_summaries
from application.invoices.institution_invoice_eligibility import (
    invoice_gate_status,
    resolve_commercial_origin,
)
from ext import db
from models import BillingParty, Booking, Company, InstitutionPatient, TransportRequest


def _segment_type(booking: Booking) -> str:
    if bool(getattr(booking, "is_return", False)):
        return "return"
    parent_id = getattr(booking, "parent_booking_id", None)
    if parent_id is not None:
        return "segment"
    route_group = getattr(booking, "route_group_id", None)
    if route_group:
        return "segment"
    return "outbound"


def _patient_display(booking: Booking) -> str | None:
    ip: InstitutionPatient | None = getattr(booking, "institution_patient", None)
    if ip is not None:
        parts = [p for p in (ip.first_name, ip.last_name) if p]
        if parts:
            return " ".join(parts).strip()
    name = getattr(booking, "customer_name", None)
    return str(name).strip() if name else None


def _payer_display_name(booking: Booking) -> str | None:
    bp: BillingParty | None = getattr(booking, "billing_party", None)
    if bp is not None and (bp.display_name or "").strip():
        return bp.display_name.strip()
    billed_to = (getattr(booking, "billed_to_type", None) or "").lower()
    if billed_to == "clinic":
        cid = getattr(booking, "billed_to_company_id", None)
        if cid is not None:
            company = db.session.get(Company, int(cid))
            if company and company.name:
                return company.name
    return _patient_display(booking)


def _transport_company_display(booking: Booking) -> dict[str, Any] | None:
    company: Company | None = getattr(booking, "company", None)
    cid = getattr(booking, "company_id", None)
    if company is None and cid is not None:
        company = db.session.get(Company, int(cid))
    if company is None:
        return None
    return {
        "company_id": int(company.id),
        "display_name": company.name,
    }


def _billing_block(booking: Booking) -> dict[str, bool]:
    locked, _ = booking_billing_is_locked(booking)
    invoiced = bool(getattr(booking, "invoice_line_id", None))
    return {
        "invoiced": invoiced,
        "locked": locked,
        "editable": not locked,
    }


def _resolve_siblings(
    booking: Booking,
    *,
    institution_bookings_by_id: dict[int, Booking],
) -> list[dict[str, Any]]:
    siblings: list[dict[str, Any]] = []
    bid = int(booking.id)

    parent_id = getattr(booking, "parent_booking_id", None)
    if parent_id is not None:
        try:
            pid = int(parent_id)
            parent = institution_bookings_by_id.get(pid)
            if parent is not None and int(parent.id) != bid:
                siblings.append(_sibling_ref(parent))
        except (TypeError, ValueError):
            pass

    for other in institution_bookings_by_id.values():
        if int(other.id) == bid:
            continue
        opid = getattr(other, "parent_booking_id", None)
        if opid is not None:
            try:
                if int(opid) == bid:
                    siblings.append(_sibling_ref(other))
            except (TypeError, ValueError):
                pass

    route_group = getattr(booking, "route_group_id", None)
    if route_group:
        for other in institution_bookings_by_id.values():
            if int(other.id) == bid:
                continue
            if getattr(other, "route_group_id", None) == route_group:
                siblings.append(_sibling_ref(other))

    seen: set[int] = set()
    unique: list[dict[str, Any]] = []
    for s in siblings:
        sid = int(s["booking_id"])
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(s)
    return sorted(unique, key=lambda x: int(x["booking_id"]))


def _sibling_ref(booking: Booking) -> dict[str, Any]:
    return {
        "booking_id": int(booking.id),
        "segment_type": _segment_type(booking),
        "effective_status": effective_control_status(booking),
    }


def _dispute_control_fields(
    booking: Booking, *, dispute_summary: dict[str, Any] | None = None
) -> dict[str, Any]:
    extra = dispute_summary
    if extra is None:
        extra = latest_dispute_summaries([int(booking.id)]).get(int(booking.id), {})
    return {
        "dispute_id": extra.get("dispute_id"),
        "dispute_status": extra.get("dispute_status"),
        "dispute_treatable": bool(extra.get("dispute_treatable")),
    }


def serialize_billing_control_booking(
    booking: Booking,
    *,
    transport_request: TransportRequest | None = None,
    institution_bookings_by_id: dict[int, Booking] | None = None,
    dispute_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Représentation API exploitable par la future UI Marc."""
    control = control_status_snapshot(booking)
    scheduled = getattr(booking, "scheduled_time", None)
    st_iso = scheduled.isoformat() if scheduled is not None else None
    siblings_ctx = institution_bookings_by_id or {int(booking.id): booking}

    return {
        "booking_id": int(booking.id),
        "request_id": int(transport_request.id)
        if transport_request is not None
        else None,
        "parent_booking_id": getattr(booking, "parent_booking_id", None),
        "route_group_id": getattr(booking, "route_group_id", None),
        "date": st_iso,
        "scheduled_time": st_iso,
        "patient": {
            "institution_patient_id": getattr(booking, "institution_patient_id", None),
            "display_name": _patient_display(booking),
        },
        "pickup": getattr(booking, "pickup_location", None),
        "dropoff": getattr(booking, "dropoff_location", None),
        "segment_type": _segment_type(booking),
        "transport_company": _transport_company_display(booking),
        "payer": {
            "type": getattr(booking, "billed_to_type", None),
            "display_name": _payer_display_name(booking),
            "billed_to_company_id": getattr(booking, "billed_to_company_id", None),
            "billing_party_id": getattr(booking, "billing_party_id", None),
        },
        "control": {
            "effective_status": control["control_status"],
            "invoice_gate_status": invoice_gate_status(booking),
            "commercial_origin": resolve_commercial_origin(booking),
            "validated_at": control.get("validated_at"),
            "validated_by_display_name": control.get("validated_by_display_name"),
            "anomaly_reason": control.get("anomaly_reason"),
            "invoice_billing_status": getattr(booking, "invoice_billing_status", None),
            **_dispute_control_fields(booking, dispute_summary=dispute_summary),
        },
        "billing": _billing_block(booking),
        "relationship": {
            "parent_booking_id": getattr(booking, "parent_booking_id", None),
            "route_group_id": getattr(booking, "route_group_id", None),
            "segment_type": _segment_type(booking),
            "siblings": _resolve_siblings(
                booking, institution_bookings_by_id=siblings_ctx
            ),
        },
        # Compat routes existantes (C01–C18)
        "control_status": control["control_status"],
        "billed_to_type": getattr(booking, "billed_to_type", None),
        "billing_party_id": getattr(booking, "billing_party_id", None),
        "billed_to_company_id": getattr(booking, "billed_to_company_id", None),
        "is_return": bool(getattr(booking, "is_return", False)),
        "billing_locked": _billing_block(booking)["locked"],
    }

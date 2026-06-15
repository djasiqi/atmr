"""Agrégation des données pour export PDF mission institution (Bon + Rapport).

Lecture seule — fallbacks « — » pour toute donnée absente (STOP GATE PDF-01).
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any

from models.booking_message import BookingMessage
from services.institutions.transport_request_display import (
    build_transport_request_display_blocks,
)
from services.institutions.transport_timeline_service import (
    build_timeline_label,
    list_timeline_events,
)

if TYPE_CHECKING:
    from models import Booking, Institution, TransportRequest

MISSING = "—"
MAX_TIMELINE_EVENTS = 500
MAX_MESSAGES = 200
MAX_ROUTE_LEGS = 20

_MISSION_TYPE_LABELS = {
    "patient_transport": "Transport patient",
    "material_delivery": "Livraison matériel",
}

_BILLING_INTENT_LABELS = {
    "patient": "Patient",
    "institution": "Institution",
    "clinic": "Institution",
    "curator": "Curateur",
    "spc": "SPC",
    "insurance": "Assurance",
    "other": "Autre",
}

_CLIENT_CATEGORY_LABELS = {
    "institution_patient": "Patient institutionnel",
    "partner_client": "Client entreprise partenaire",
    "lirie_client": "Client plateforme LIRIE",
    "lirie_guest": "Client plateforme LIRIE",
    "company_client": "Client entreprise transport",
}

_MOBILITY_LABELS = {
    "wheelchair": "Fauteuil roulant",
    "vehicle_wheelchair": "Fauteuil véhicule",
    "stretcher": "Brancard",
    "needs_assistance": "Accompagnement",
    "oxygen": "Oxygène",
    "walking": "Transport assis",
}


@dataclass
class MissionReportContext:
    reference: str
    request_number: str
    booking_number: str | None
    status_label: str
    generated_at: datetime
    variant: str

    request_classification: dict[str, Any]
    client_identity: dict[str, Any]
    patient_block: dict[str, Any]
    institution_snapshot: dict[str, Any]
    carrier_block: dict[str, Any]

    mission_info: dict[str, Any]
    billing_block: dict[str, Any]
    medical_block: dict[str, Any]
    route_steps: list[dict[str, Any]]
    mission_milestones: list[dict[str, Any]]
    synthetic_history: list[dict[str, Any]]

    timeline_rows: list[dict[str, Any]]
    timeline_truncated: bool = False
    messages: list[dict[str, Any]] = field(default_factory=list)
    messages_truncated: bool = False
    route_legs_truncated: bool = False

    attachments: list[dict[str, Any]] = field(default_factory=list)
    traceability: dict[str, Any] = field(default_factory=dict)
    completion_certificate: dict[str, Any] | None = None
    gps_proof: dict[str, Any] | None = None

    show_amount: bool = True


def format_transport_reference(tr: TransportRequest) -> str:
    year = tr.created_at.year if tr.created_at else datetime.now(UTC).year
    return f"TR-{year}-{tr.id:06d}"


def _slugify_filename_part(text: str, *, max_len: int = 48) -> str:
    """Normalise un libellé pour un nom de fichier sûr (ASCII, tirets)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", ascii_text).strip("-")
    return slug[:max_len] if slug else ""


def _patient_filename_slug(tr: TransportRequest) -> str:
    patient = getattr(tr, "patient", None)
    if patient is not None:
        last = _slugify_filename_part(getattr(patient, "last_name", "") or "", max_len=28)
        first = _slugify_filename_part(getattr(patient, "first_name", "") or "", max_len=18)
        if last and first:
            return f"{last}-{first}"
        if last or first:
            return last or first
    ext_ref = (getattr(tr, "external_reference", None) or "").strip()
    if ext_ref:
        return _slugify_filename_part(ext_ref, max_len=36) or "Sans-patient"
    return "Sans-patient"


def build_mission_pdf_filename(
    tr: TransportRequest,
    *,
    variant: str = "audit",
    disambiguate: bool = False,
) -> str:
    """Nom de fichier PDF : date + patient + type de document (id seulement si doublon).

    Exemple audit : ``2026-06-14_STOFER-THOMI-Eliane_Rapport-mission.pdf``
    Exemple bon   : ``2026-06-14_STOFER-THOMI-Eliane_Bon-transport.pdf``
    """
    doc_label = "Bon-transport" if variant == "operational" else "Rapport-mission"
    mission = getattr(tr, "mission_date", None)
    date_part = mission.isoformat() if isinstance(mission, date) else "sans-date"
    patient_part = _patient_filename_slug(tr)
    base = f"{date_part}_{patient_part}_{doc_label}"
    if disambiguate:
        return f"{base}_{tr.id}.pdf"
    return f"{base}.pdf"


def make_unique_mission_pdf_filenames(
    requests: list[TransportRequest],
    *,
    variant: str = "audit",
) -> dict[int, str]:
    """Attribue un nom unique par demande (suffixe id si collision même jour/patient)."""
    seen: dict[str, int] = {}
    out: dict[int, str] = {}
    for tr in requests:
        mission = getattr(tr, "mission_date", None)
        date_part = mission.isoformat() if isinstance(mission, date) else "sans-date"
        patient_part = _patient_filename_slug(tr)
        doc_label = "Bon-transport" if variant == "operational" else "Rapport-mission"
        key = f"{date_part}|{patient_part}|{doc_label}".lower()
        seen[key] = seen.get(key, 0) + 1
        out[tr.id] = build_mission_pdf_filename(
            tr,
            variant=variant,
            disambiguate=seen[key] > 1,
        )
    return out


def format_request_number(tr: TransportRequest) -> str:
    return f"#{tr.id}"


def format_booking_number(booking: Booking | None) -> str | None:
    if booking is None:
        return None
    return f"#{booking.id}"


def _status_str(raw: Any) -> str:
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw) or "").upper()


def build_mission_status_label(tr: TransportRequest, booking: Booking | None) -> str:
    """Libellé métier FR — jamais de code technique brut."""
    from models.enums import RequestStatus

    if booking is not None:
        b_status = _status_str(booking.status)
        if b_status == "COMPLETED":
            if bool(getattr(tr, "is_round_trip", False)):
                return "Réalisé (aller-retour)"
            return "Réalisé"
        if b_status == "RETURN_COMPLETED":
            return "Réalisé (aller-retour)"
        if b_status == "CANCELED":
            return "Annulé"
        if b_status in {"EN_ROUTE", "IN_PROGRESS", "ASSIGNED", "ACCEPTED", "PENDING"}:
            return "En cours de réalisation"
        if b_status == "AWAITING_CLIENT_PAYMENT":
            return "En attente de paiement"

    req_status = _status_str(getattr(tr, "status", None))
    return RequestStatus.display_label(req_status) if req_status else MISSING


def _carrier_source_value(tr: TransportRequest) -> str:
    from models.enums import CarrierSource

    raw = getattr(tr, "carrier_source", None) or CarrierSource.LIRIE.value
    return str(getattr(raw, "value", raw) or CarrierSource.LIRIE.value)


def _format_user_name(user: Any) -> str:
    if not user:
        return MISSING
    first = getattr(user, "first_name", "") or ""
    last = getattr(user, "last_name", "") or ""
    name = f"{first} {last}".strip()
    return name or getattr(user, "username", None) or MISSING


def _carrier_block_defaults(
    *,
    name: str,
    phone: str,
    email: str,
    driver_name: str,
    driver_phone: str,
    vehicle: str,
    is_external: bool,
    execution_mode_label: str,
    externalization_reason: str = MISSING,
    reference: str = MISSING,
    assigned_externally_at: str = MISSING,
    externalized_by_name: str = MISSING,
    declared_at: str = MISSING,
    declared_by: str = MISSING,
    execution_notes: str = MISSING,
) -> dict[str, Any]:
    return {
        "name": name,
        "phone": phone,
        "email": email,
        "driver_name": driver_name,
        "driver_phone": driver_phone,
        "vehicle": vehicle,
        "is_external": is_external,
        "execution_mode_label": execution_mode_label,
        "externalization_reason": externalization_reason,
        "reference": reference,
        "assigned_externally_at": assigned_externally_at,
        "externalized_by_name": externalized_by_name,
        "declared_at": declared_at,
        "declared_by": declared_by,
        "execution_notes": execution_notes,
    }


def build_carrier_block(tr: TransportRequest, booking: Booking | None) -> dict[str, Any]:
    from models.enums import CarrierSource

    carrier_source = _carrier_source_value(tr)
    execution_mode_label = CarrierSource.display_label(carrier_source)
    is_external = carrier_source == CarrierSource.EXTERNAL.value

    if is_external:
        return _carrier_block_defaults(
            name=getattr(tr, "external_carrier_name", None) or "Transporteur externe",
            phone=getattr(tr, "external_carrier_phone", None) or MISSING,
            email=MISSING,
            driver_name=MISSING,
            driver_phone=MISSING,
            vehicle=MISSING,
            is_external=True,
            execution_mode_label=execution_mode_label,
            externalization_reason=getattr(tr, "external_carrier_reason", None) or MISSING,
            reference=getattr(tr, "external_carrier_reference", None) or MISSING,
            assigned_externally_at=_fmt_dt(getattr(tr, "assigned_externally_at", None)),
            externalized_by_name=_format_user_name(getattr(tr, "externalized_by", None)),
            declared_at=_fmt_dt(getattr(tr, "executed_externally_at", None)),
            declared_by=_format_user_name(getattr(tr, "executed_externally_by", None)),
            execution_notes=getattr(tr, "external_execution_notes", None) or MISSING,
        )

    company = getattr(tr, "accepted_by_company", None)
    if company is None and booking is not None:
        company = getattr(booking, "company", None)
    driver_name, driver_phone, vehicle = _driver_details(booking)
    if company is None:
        return _carrier_block_defaults(
            name="Non assignée",
            phone=MISSING,
            email=MISSING,
            driver_name=driver_name,
            driver_phone=driver_phone,
            vehicle=vehicle,
            is_external=False,
            execution_mode_label=execution_mode_label,
        )
    return _carrier_block_defaults(
        name=getattr(company, "name", None) or MISSING,
        phone=getattr(company, "contact_phone", None) or MISSING,
        email=getattr(company, "contact_email", None) or MISSING,
        driver_name=driver_name,
        driver_phone=driver_phone,
        vehicle=vehicle,
        is_external=False,
        execution_mode_label=execution_mode_label,
    )


def _fmt_dt(value: Any, fmt: str = "%d.%m.%Y %H:%M") -> str:
    if isinstance(value, datetime):
        return value.strftime(fmt)
    if isinstance(value, date):
        return value.strftime("%d.%m.%Y")
    return MISSING


def _fmt_iso(value: str | None, fmt: str = "%d.%m.%Y %H:%M") -> str:
    if not value:
        return MISSING
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime(fmt)
    except ValueError:
        return MISSING


def build_institution_snapshot(tr: TransportRequest, institution: Institution) -> dict[str, Any]:
    contact = getattr(tr, "contact_on_site", None) or {}
    if not isinstance(contact, dict):
        contact = {}
    requester = tr._get_creator_name() if hasattr(tr, "_get_creator_name") else None
    return {
        "name": getattr(institution, "name", None) or MISSING,
        "contact_phone": getattr(institution, "contact_phone", None) or MISSING,
        "contact_email": getattr(institution, "contact_email", None) or MISSING,
        "service": contact.get("requester_service") or MISSING,
        "requester_name": requester or contact.get("requester_name") or MISSING,
        "requester_phone": contact.get("requester_phone") or contact.get("phone") or MISSING,
        "logo_url": getattr(institution, "logo_url", None),
        "captured_at": datetime.now(UTC).isoformat(),
    }


def resolve_institution_snapshot(tr: TransportRequest, institution: Institution) -> dict[str, Any]:
    """Snapshot figé : priorité au payload `request_converted` (V2), sinon construction live."""
    try:
        from services.institutions.transport_timeline_service import find_latest_event

        ev = find_latest_event(
            transport_request_id=tr.id,
            event_type="request_converted",
        )
        payload = getattr(ev, "payload", None) if ev else None
        if isinstance(payload, dict) and payload.get("institution_snapshot"):
            snap = dict(payload["institution_snapshot"])
            snap["source"] = "persisted"
            return snap
    except Exception:
        pass
    snap = build_institution_snapshot(tr, institution)
    snap["source"] = "generated"
    return snap


def build_client_identity_block(tr: TransportRequest, booking: Booking | None) -> dict[str, Any]:
    display = build_transport_request_display_blocks(tr)
    identity = display.get("identity") or {}
    category = identity.get("display_category") or "institution_patient"
    source = identity.get("source") or {}
    passenger = (identity.get("passenger") or {}).get("name") or MISSING
    org_name = source.get("name") or (getattr(tr.institution, "name", None) if getattr(tr, "institution", None) else MISSING)

    if category == "institution_patient":
        headline = f"Patient : {passenger}"
        subline = f"Institution : {org_name}"
    elif category == "partner_client":
        headline = f"Client : {passenger}"
        subline = f"Entreprise partenaire : {org_name}"
    elif category in {"lirie_client", "lirie_guest"}:
        headline = "Client plateforme LIRIE"
        subline = passenger if passenger != MISSING else MISSING
    else:
        headline = f"Client : {passenger}"
        subline = "Client entreprise transport"

    return {
        "display_category": category,
        "category_label": _CLIENT_CATEGORY_LABELS.get(category, category),
        "passenger_name": passenger,
        "organization_name": org_name,
        "headline": headline,
        "subline": subline,
    }


def build_patient_block(tr: TransportRequest) -> dict[str, Any]:
    patient = getattr(tr, "patient", None)
    if patient is None:
        name = getattr(tr, "external_reference", None) or f"Demande {tr.id}"
        return {
            "last_name": name,
            "first_name": MISSING,
            "full_name": name,
            "dob": MISSING,
            "dossier_number": MISSING,
            "room": MISSING,
            "address": MISSING,
        }
    dob = getattr(patient, "dob", None)
    room_parts: list[str] = []
    floor = getattr(patient, "floor", None)
    residence = getattr(patient, "residence_name", None)
    if floor:
        room_parts.append(f"Étage {floor}")
    if residence:
        room_parts.append(str(residence))
    room_label = " · ".join(room_parts) if room_parts else MISSING
    return {
        "last_name": patient.last_name or MISSING,
        "first_name": patient.first_name or MISSING,
        "full_name": f"{patient.last_name} {patient.first_name}".strip() or MISSING,
        "dob": patient.dob.strftime("%d.%m.%Y") if dob else MISSING,
        "dossier_number": getattr(patient, "external_reference", None) or MISSING,
        "room": room_label,
        "address": _format_patient_address(patient),
    }


def _format_patient_address(patient: Any) -> str:
    """Adresse patient « rue, NPA ville » (identification facturation patient)."""
    street = (getattr(patient, "address", None) or "").strip()
    postal = (getattr(patient, "postal_code", None) or "").strip()
    city = (getattr(patient, "city", None) or "").strip()
    locality = " ".join(part for part in (postal, city) if part).strip()
    parts = [part for part in (street, locality) if part]
    return ", ".join(parts) if parts else MISSING


def _driver_details(booking: Booking | None) -> tuple[str, str, str]:
    """Retourne (nom chauffeur, téléphone pro, véhicule)."""
    if booking is None:
        return MISSING, MISSING, MISSING
    driver = getattr(booking, "driver", None)
    if driver is None:
        return MISSING, MISSING, MISSING
    user = getattr(driver, "user", None)
    if user:
        name = f"{getattr(user, 'first_name', '') or ''} {getattr(user, 'last_name', '') or ''}".strip()
        name = name or getattr(user, "username", None) or MISSING
        phone = getattr(user, "phone", None) or MISSING
    else:
        name = MISSING
        phone = MISSING
    vehicle = getattr(driver, "vehicle_assigned", None) or MISSING
    return name, phone, vehicle


def _derive_mobility_level(mobility: dict[str, Any]) -> str:
    if mobility.get("stretcher"):
        return "stretcher"
    if mobility.get("wheelchair") or mobility.get("vehicle_wheelchair"):
        return "wheelchair"
    if mobility.get("needs_assistance"):
        return "assisted"
    return "ambulatory"


def _resolve_billing_target(tr: TransportRequest, booking: Booking | None) -> str:
    if booking is not None:
        raw = getattr(booking, "billed_to_type", None)
        key = str(getattr(raw, "value", raw) or "patient").lower()
        if key == "clinic":
            return "institution"
        return key
    intent = getattr(tr, "billing_intent", None) or "patient"
    return str(intent).lower()


def build_request_classification(tr: TransportRequest, booking: Booking | None) -> dict[str, Any]:
    display = build_transport_request_display_blocks(tr)
    trip_flags = display.get("trip_flags") or {}
    mobility = tr.get_mobility() if hasattr(tr, "get_mobility") else {}
    if trip_flags.get("round_trip"):
        trip_type = "round_trip"
    elif trip_flags.get("multi_stop"):
        trip_type = "multi_stop"
    else:
        trip_type = "one_way"
    identity = display.get("identity") or {}
    return {
        "trip_type": trip_type,
        "transport_type": getattr(tr, "mission_type", None) or "patient_transport",
        "mobility_level": _derive_mobility_level(mobility),
        "scheduled_time_type": getattr(tr, "scheduled_time_type", None) or "departure",
        "billing_target": _resolve_billing_target(tr, booking),
        "return_to_institution": bool(trip_flags.get("return_to_institution")),
        "display_category": identity.get("display_category"),
        "carrier_source": _carrier_source_value(tr),
    }


def build_medical_block(tr: TransportRequest) -> dict[str, Any]:
    mobility = tr.get_mobility() if hasattr(tr, "get_mobility") else {}
    lines: list[dict[str, str]] = []
    for key, label in _MOBILITY_LABELS.items():
        if key not in mobility:
            continue
        val = mobility[key]
        if isinstance(val, bool):
            display = "Oui" if val else "Non"
        else:
            display = str(val) if val else MISSING
        lines.append({"label": label, "value": display})
    notes = getattr(tr, "notes", None)
    floor_info = getattr(tr, "floor_elevator_info", None)
    return {
        "lines": lines,
        "notes": (notes or "").strip() or MISSING,
        "floor_elevator_info": (floor_info or "").strip() or MISSING,
    }


def build_billing_block(
    tr: TransportRequest,
    booking: Booking | None,
    *,
    show_amount: bool = True,
) -> dict[str, Any]:
    target_key = _resolve_billing_target(tr, booking)
    billed_label = _BILLING_INTENT_LABELS.get(target_key, target_key or MISSING)
    from models.enums import CarrierSource

    if _carrier_source_value(tr) == CarrierSource.EXTERNAL.value:
        carrier_name = getattr(tr, "external_carrier_name", None) or MISSING
    else:
        company = getattr(tr, "accepted_by_company", None)
        carrier_name = getattr(company, "name", None) if company else MISSING

    amount_str = MISSING
    invoice_status = MISSING
    if booking is not None and show_amount:
        amt = getattr(booking, "amount", None)
        if amt is not None:
            amount_str = f"{float(amt):.2f} CHF"
        summary = tr._serialize_booking_summary() if hasattr(tr, "_serialize_booking_summary") else None
        if summary:
            b_status = _status_str(booking.status)
            if b_status == "CANCELED":
                if summary.get("is_cancellation_billable"):
                    invoice_status = "Annulé — facturé"
                else:
                    invoice_status = "Annulé — non facturé"
            elif summary.get("is_invoiced"):
                invoice_status = "Facturé"
            else:
                invoice_status = "À facturer"
        elif amt is not None:
            invoice_status = "À facturer"
    elif not show_amount:
        amount_str = MISSING
        invoice_status = MISSING
    else:
        intent = getattr(tr, "billing_intent", None)
        invoice_status = _BILLING_INTENT_LABELS.get(str(intent or ""), MISSING)

    return {
        "billed_to": billed_label,
        "carrier_name": carrier_name or MISSING,
        "amount": amount_str,
        "invoice_status": invoice_status,
    }


def build_mission_info(tr: TransportRequest, booking: Booking | None) -> dict[str, Any]:
    mtype = getattr(tr, "mission_type", None) or "patient_transport"
    return {
        "type_label": _MISSION_TYPE_LABELS.get(mtype, mtype),
        "reference": format_transport_reference(tr),
        "created_at": _fmt_dt(getattr(tr, "created_at", None)),
        "accepted_at": _fmt_dt(getattr(tr, "accepted_at", None)),
        "mission_date": _fmt_dt(getattr(tr, "mission_date", None), "%d.%m.%Y"),
        "billing_mode": _BILLING_INTENT_LABELS.get(
            str(getattr(tr, "billing_intent", "") or ""),
            getattr(tr, "billing_intent", None) or MISSING,
        ),
    }


def _leg_time_display(leg: Any, display_legs: list[dict[str, Any]], index: int) -> str:
    if index < len(display_legs):
        return display_legs[index].get("display_time") or MISSING
    st = getattr(leg, "scheduled_time", None)
    if st and getattr(leg, "time_confirmed", False):
        return st.strftime("%H:%M")
    if st:
        return f"{st.strftime('%H:%M')} (indicatif)"
    return MISSING


def build_route_steps(
    tr: TransportRequest,
    booking: Booking | None,
    display_blocks: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Construit les étapes trajet (prévu / réel). Retourne (steps, truncated)."""
    truncated = False
    scheduling = display_blocks.get("scheduling") or {}
    departure = scheduling.get("departure") or {}
    display_legs = display_blocks.get("legs") or []

    route_journey: list[dict[str, Any]] = []
    if booking is not None:
        try:
            route_journey = booking._get_route_journey() or []
        except Exception:
            route_journey = []

    pickup_events = [e for e in route_journey if e.get("type") == "pickup"]
    dropoff_events = [e for e in route_journey if e.get("type") == "dropoff"]

    boarded_at = MISSING
    if booking is not None and getattr(booking, "boarded_at", None):
        boarded_at = _fmt_dt(booking.boarded_at, "%H:%M")
    elif pickup_events:
        boarded_at = _fmt_iso(pickup_events[0].get("date"), "%H:%M")

    steps: list[dict[str, Any]] = []

    legs = sorted(
        list(getattr(tr, "legs", None) or []),
        key=lambda lg: getattr(lg, "sequence_index", 0),
    )
    if len(legs) > MAX_ROUTE_LEGS:
        legs = legs[:MAX_ROUTE_LEGS]
        truncated = True

    if legs:
        first = legs[0]
        steps.append(
            {
                "kind": "departure",
                "title": "Départ",
                "address": getattr(first, "pickup_location", None) or tr.pickup_location or MISSING,
                "planned_time": departure.get("display_time") or MISSING,
                "actual_time": boarded_at,
            }
        )
        for idx, leg in enumerate(legs):
            est = getattr(leg, "dropoff_establishment", None)
            addr = getattr(leg, "dropoff_location", None) or MISSING
            label = f"Destination {idx + 1}"
            if est:
                label = est
            actual = MISSING
            if idx < len(dropoff_events):
                actual = _fmt_iso(dropoff_events[idx].get("date"), "%H:%M")
            steps.append(
                {
                    "kind": "destination",
                    "title": label,
                    "address": addr,
                    "planned_time": _leg_time_display(leg, display_legs, idx),
                    "actual_time": actual,
                }
            )
        if bool(getattr(tr, "is_round_trip", False)):
            ret_sched = scheduling.get("return") or {}
            ret_actual = MISSING
            if dropoff_events:
                ret_actual = _fmt_iso(dropoff_events[-1].get("date"), "%H:%M")
            elif booking is not None and getattr(booking, "completed_at", None):
                ret_actual = _fmt_dt(booking.completed_at, "%H:%M")
            ret_addr = tr.pickup_location or MISSING
            if bool(getattr(tr, "return_to_institution", False)):
                inst = getattr(tr, "institution", None)
                if inst and getattr(inst, "address", None):
                    ret_addr = inst.address
            steps.append(
                {
                    "kind": "return",
                    "title": "Retour",
                    "address": ret_addr,
                    "planned_time": ret_sched.get("display_time") or MISSING,
                    "actual_time": ret_actual,
                }
            )
    else:
        steps.append(
            {
                "kind": "departure",
                "title": "Départ",
                "address": tr.pickup_location or MISSING,
                "planned_time": departure.get("display_time") or MISSING,
                "actual_time": boarded_at,
            }
        )
        steps.append(
            {
                "kind": "destination",
                "title": "Destination",
                "address": tr.dropoff_location or MISSING,
                "planned_time": MISSING,
                "actual_time": _fmt_iso(dropoff_events[0].get("date"), "%H:%M")
                if dropoff_events
                else (
                    _fmt_dt(booking.completed_at, "%H:%M")
                    if booking and getattr(booking, "completed_at", None)
                    else MISSING
                ),
            }
        )
        if bool(getattr(tr, "is_round_trip", False)):
            ret_sched = scheduling.get("return") or {}
            ret_actual = MISSING
            if dropoff_events:
                ret_actual = _fmt_iso(dropoff_events[-1].get("date"), "%H:%M")
            elif booking is not None and getattr(booking, "completed_at", None):
                ret_actual = _fmt_dt(booking.completed_at, "%H:%M")
            ret_addr = tr.pickup_location or MISSING
            if bool(getattr(tr, "return_to_institution", False)):
                inst = getattr(tr, "institution", None)
                if inst and getattr(inst, "address", None):
                    ret_addr = inst.address
            steps.append(
                {
                    "kind": "return",
                    "title": "Retour",
                    "address": ret_addr,
                    "planned_time": ret_sched.get("display_time") or MISSING,
                    "actual_time": ret_actual,
                }
            )

    return steps, truncated


def resolve_timeline_actor(event: Any, institution: Institution, tr: TransportRequest) -> str:
    payload = getattr(event, "payload", None) or {}
    actor_type = getattr(event, "actor_type", "") or ""

    if actor_type == "system":
        return "LIRIE"
    if actor_type in {"company", "company_user"}:
        return payload.get("company_name") or "Transporteur"
    if actor_type == "driver":
        return payload.get("driver_name") or "Chauffeur"
    if actor_type == "institution_user":
        if payload.get("actor_name"):
            return str(payload["actor_name"])
        uid = getattr(event, "actor_user_id", None)
        if uid:
            try:
                from models import User

                user = User.query.get(uid)
                if user:
                    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
                    return name or user.username or institution.name
            except Exception:
                pass
        return institution.name
    if actor_type == "api_key":
        return institution.name
    return institution.name


def resolve_timeline_channel(event: Any) -> str:
    actor_type = getattr(event, "actor_type", "") or ""
    mapping = {
        "system": "Automatique",
        "institution_user": "Portail institution",
        "api_key": "API / Intégration",
        "company": "Portail transporteur",
        "company_user": "Portail transporteur",
        "driver": "Mobile chauffeur",
    }
    return mapping.get(actor_type, "Système")


def build_mission_milestones(
    tr: TransportRequest,
    booking: Booking | None,
    timeline_events: list[Any],
) -> list[dict[str, Any]]:
    """Jalons opérationnels dérivés de la timeline et des timestamps booking (V1.1)."""
    raw: list[tuple[datetime, str, str]] = []

    def _add(dt: Any, label: str, source: str) -> None:
        if not isinstance(dt, datetime):
            return
        raw.append((dt, label, source))

    for ev in timeline_events:
        et = getattr(ev, "event_type", "") or ""
        payload = getattr(ev, "payload", None) or {}
        created = getattr(ev, "created_at", None)
        if et == "offer_accepted":
            _add(created, "Acceptation transporteur", "timeline")
        elif et == "driver_assigned":
            name = payload.get("driver_name") or "Chauffeur"
            _add(created, f"Chauffeur assigné ({name})", "timeline")
        elif et == "patient_boarded":
            _add(created, "Patient embarqué", "timeline")
        elif et == "patient_completed":
            _add(created, "Mission terminée", "timeline")
        elif et == "external_carrier_switched":
            _add(created, "Mission basculée vers transporteur externe", "timeline")
        elif et == "external_carrier_assigned":
            name = payload.get("carrier_name") or "Transporteur externe"
            _add(created, f"Transporteur externe affecté : {name}", "timeline")
        elif et == "external_mission_completed":
            _add(created, "Déclarée réalisée par transporteur externe", "timeline")
        elif et == "status_changed":
            new_s = str(payload.get("new_status") or "").upper()
            if new_s == "EN_ROUTE":
                _add(created, "En route", "timeline")
            elif new_s in {"IN_PROGRESS", "PICKED_UP"}:
                _add(created, "Arrivé sur site", "timeline")
            elif new_s in {"COMPLETED", "RETURN_COMPLETED"}:
                _add(created, "Course terminée", "timeline")

    if getattr(tr, "accepted_at", None):
        _add(tr.accepted_at, "Acceptation enregistrée", "demande")
    if getattr(tr, "assigned_externally_at", None):
        _add(tr.assigned_externally_at, "Bascule transporteur externe", "demande")
    if getattr(tr, "executed_externally_at", None):
        _add(tr.executed_externally_at, "Déclarée réalisée par l'institution", "demande")
    if booking is not None:
        if getattr(booking, "boarded_at", None):
            _add(booking.boarded_at, "Prise en charge", "réservation")
        if getattr(booking, "completed_at", None):
            _add(booking.completed_at, "Fin de course", "réservation")

    raw.sort(key=lambda item: item[0])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for dt, label, source in raw:
        dedupe_key = f"{label}|{dt.replace(microsecond=0).isoformat()}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        rows.append(
            {
                "datetime": _fmt_dt(dt, "%d.%m.%Y %H:%M"),
                "milestone": label,
                "source": source,
            }
        )
    return rows


MAX_SYNTHETIC_HISTORY_STANDARD = 4
MAX_SYNTHETIC_HISTORY_ENRICHED = 6

_INTERNAL_TIMELINE_EVENTS = frozenset(
    {
        "offer_sent",
        "request_converted",
        "booking_created",
        "field_updated",
        "driver_assigned",
        "driver_reassigned",
        "status_changed",
    }
)


def build_synthetic_history(
    tr: TransportRequest,
    booking: Booking | None,
    timeline_events: list[Any],
    *,
    enriched: bool = False,
) -> list[dict[str, Any]]:
    """Historique institutionnel synthétique (4 lignes standard, 6 max enrichi)."""
    max_rows = MAX_SYNTHETIC_HISTORY_ENRICHED if enriched else MAX_SYNTHETIC_HISTORY_STANDARD
    raw: list[tuple[datetime, str]] = []

    def _is_cancelled() -> bool:
        if booking is not None and _status_str(booking.status) == "CANCELED":
            return True
        req_status = _status_str(getattr(tr, "status", None))
        return req_status in {"CANCELLED", "CANCELED"}

    cancelled = _is_cancelled()

    def _add(dt: Any, label: str) -> None:
        if not isinstance(dt, datetime):
            return
        raw.append((dt, label))

    for ev in timeline_events:
        et = getattr(ev, "event_type", "") or ""
        if et in _INTERNAL_TIMELINE_EVENTS:
            continue
        payload = getattr(ev, "payload", None) or {}
        created = getattr(ev, "created_at", None)
        if et == "request_created":
            _add(created, "Demande créée")
        elif et == "offer_accepted":
            name = payload.get("company_name") or "transporteur"
            _add(created, f"Acceptée par {name}")
        elif et == "external_carrier_switched":
            _add(created, "Mission basculée vers transporteur externe")
        elif et == "external_carrier_assigned":
            name = payload.get("carrier_name") or "Transporteur externe"
            _add(created, f"Transporteur externe affecté : {name}")
        elif et == "external_mission_completed" and not cancelled:
            _add(created, "Déclarée réalisée par transporteur externe")
        elif et == "patient_boarded" and not cancelled:
            _add(created, "Prise en charge")
        elif et == "patient_completed" and not cancelled:
            _add(created, "Mission terminée")
        elif et == "cancelled":
            _add(created, "Annulée")

    has_created = any(label == "Demande créée" for _, label in raw)
    if not has_created and getattr(tr, "created_at", None):
        _add(tr.created_at, "Demande créée")

    has_accepted = any(label.startswith("Acceptée par") for _, label in raw)
    if not has_accepted and getattr(tr, "accepted_at", None):
        company = getattr(getattr(tr, "accepted_by_company", None), "name", None)
        if company:
            _add(tr.accepted_at, f"Acceptée par {company}")

    if booking is not None and not cancelled:
        has_boarded = any(label == "Prise en charge" for _, label in raw)
        if not has_boarded and getattr(booking, "boarded_at", None):
            _add(booking.boarded_at, "Prise en charge")
        has_completed = any(label == "Mission terminée" for _, label in raw)
        if not has_completed and getattr(booking, "completed_at", None):
            _add(booking.completed_at, "Mission terminée")

    from models.enums import CarrierSource, RequestStatus

    if (
        _carrier_source_value(tr) == CarrierSource.EXTERNAL.value
        and not cancelled
    ):
        if not any(
            label.startswith("Transporteur externe affecté")
            or label == "Mission basculée vers transporteur externe"
            for _, label in raw
        ):
            if getattr(tr, "assigned_externally_at", None):
                name = getattr(tr, "external_carrier_name", None) or "Transporteur externe"
                _add(tr.assigned_externally_at, f"Transporteur externe affecté : {name}")
        if (
            _status_str(getattr(tr, "status", None))
            == RequestStatus.EXTERNAL_DECLARED_COMPLETED.value
            and not any(
                label == "Déclarée réalisée par transporteur externe" for _, label in raw
            )
            and getattr(tr, "executed_externally_at", None)
        ):
            _add(tr.executed_externally_at, "Déclarée réalisée par transporteur externe")

    if cancelled and not any(label == "Annulée" for _, label in raw):
        fallback_dt = getattr(booking, "updated_at", None) if booking else None
        if not isinstance(fallback_dt, datetime):
            # Annulation = dernier événement : la placer après les autres si aucune
            # date fiable, pour garantir un historique chronologiquement cohérent.
            existing = [dt for dt, _ in raw]
            fallback_dt = max(existing) if existing else getattr(tr, "created_at", None)
        _add(fallback_dt, "Annulée")

    raw.sort(key=lambda item: item[0])
    seen_labels: set[str] = set()
    rows: list[dict[str, Any]] = []
    for dt, label in raw:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        rows.append(
            {
                "date": _fmt_dt(dt, "%d.%m.%Y %H:%M"),
                "label": label,
                "at": dt,
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def build_completion_certificate(
    *,
    reference: str,
    status_label: str,
    patient_block: dict[str, Any],
    institution_snapshot: dict[str, Any],
    carrier_block: dict[str, Any],
    mission_info: dict[str, Any],
    document_hash: str,
    public_id: str,
    generated_at: datetime,
) -> dict[str, Any] | None:
    """Certificat de réalisation (V2) — missions terminées uniquement."""
    if status_label not in {"Réalisé", "Réalisé (aller-retour)"}:
        return None
    return {
        "title": "Certificat de réalisation",
        "reference": reference,
        "status_label": status_label,
        "patient_name": patient_block.get("full_name", MISSING),
        "institution_name": institution_snapshot.get("name", MISSING),
        "carrier_name": carrier_block.get("name", MISSING),
        "mission_date": mission_info.get("mission_date", MISSING),
        "public_id": public_id,
        "document_hash": document_hash,
        "issued_at": generated_at.strftime("%d.%m.%Y %H:%M"),
    }


def build_gps_proof(booking: Booking | None) -> dict[str, Any]:
    """Preuve GPS (V3) — dernière position connue ou message de fallback."""
    fallback = "Preuve GPS disponible dans le système LIRIE"
    driver_id = getattr(booking, "driver_id", None) if booking else None
    if not driver_id:
        return {"available": False, "message": fallback}
    try:
        from infrastructure.persistence.drivers.redis_driver_location_store import (
            get_driver_last_location,
        )

        loc = get_driver_last_location(int(driver_id))
    except Exception:
        loc = None
    if not loc:
        return {"available": False, "message": fallback}
    lat = loc.get("lat")
    lon = loc.get("lon")
    if lat is None or lon is None:
        return {"available": False, "message": fallback}
    ts = loc.get("ts") or loc.get("updated_at") or loc.get("timestamp")
    ts_label = MISSING
    if isinstance(ts, str):
        ts_label = _fmt_iso(ts)
    elif isinstance(ts, (int, float)):
        try:
            ts_label = _fmt_dt(datetime.fromtimestamp(ts, tz=UTC))
        except Exception:
            ts_label = str(ts)
    elif isinstance(ts, datetime):
        ts_label = _fmt_dt(ts)
    return {
        "available": True,
        "latitude": lat,
        "longitude": lon,
        "timestamp": ts_label,
        "message": None,
    }


def collect_messages(booking_id: int | None) -> tuple[list[dict[str, Any]], bool]:
    if not booking_id:
        return [], False
    q = (
        BookingMessage.query.filter_by(booking_id=booking_id)
        .order_by(BookingMessage.created_at.asc())
    )
    total = q.count()
    truncated = total > MAX_MESSAGES
    if truncated:
        messages = q.offset(max(0, total - MAX_MESSAGES)).all()
    else:
        messages = q.all()
    rows = [
        {
            "date": _fmt_dt(m.created_at, "%d.%m.%Y %H:%M"),
            "content": m.content,
            "sender": m.sender_label,
        }
        for m in messages
    ]
    return rows, truncated


def compute_document_hash(ctx: MissionReportContext) -> str:
    generated = ctx.generated_at.replace(microsecond=0).isoformat()
    payload = {
        "public_id": ctx.traceability.get("public_id"),
        "reference": ctx.reference,
        "status": ctx.status_label,
        "generated_at": generated,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def collect_mission_report_context(
    tr: TransportRequest,
    institution: Institution,
    *,
    variant: str = "audit",
    show_amount: bool = True,
) -> MissionReportContext:
    """Agrège toutes les données pour génération PDF."""
    booking = getattr(tr, "booking", None)
    display_blocks = build_transport_request_display_blocks(tr)
    route_steps, route_truncated = build_route_steps(tr, booking, display_blocks)

    timeline_events = list_timeline_events(
        institution_id=tr.institution_id,
        transport_request_id=tr.id,
        limit=MAX_TIMELINE_EVENTS,
    )
    timeline_truncated = len(timeline_events) >= MAX_TIMELINE_EVENTS
    timeline_events_sorted = sorted(
        timeline_events,
        key=lambda e: (e.created_at or datetime.min.replace(tzinfo=UTC), e.id),
    )
    timeline_rows = [
        {
            "date": _fmt_dt(ev.created_at, "%d.%m.%Y %H:%M"),
            "action": build_timeline_label(ev),
            "actor": resolve_timeline_actor(ev, institution, tr),
            "channel": resolve_timeline_channel(ev),
        }
        for ev in timeline_events_sorted
    ]
    mission_milestones = build_mission_milestones(tr, booking, timeline_events_sorted)
    trip_type = build_request_classification(tr, booking).get("trip_type", "one_way")
    synthetic_history = build_synthetic_history(
        tr,
        booking,
        timeline_events_sorted,
        enriched=trip_type in {"multi_stop", "round_trip"},
    )

    booking_id = getattr(tr, "booking_id", None) or (booking.id if booking else None)
    messages, messages_truncated = collect_messages(booking_id)

    generated_at = datetime.now(UTC)
    institution_snapshot = resolve_institution_snapshot(tr, institution)
    patient_block = build_patient_block(tr)
    carrier_block = build_carrier_block(tr, booking)
    mission_info = build_mission_info(tr, booking)
    status_label = build_mission_status_label(tr, booking)

    ctx = MissionReportContext(
        reference=format_transport_reference(tr),
        request_number=format_request_number(tr),
        booking_number=format_booking_number(booking),
        status_label=status_label,
        generated_at=generated_at,
        variant=variant,
        request_classification=build_request_classification(tr, booking),
        client_identity=build_client_identity_block(tr, booking),
        patient_block=patient_block,
        institution_snapshot=institution_snapshot,
        carrier_block=carrier_block,
        mission_info=mission_info,
        billing_block=build_billing_block(tr, booking, show_amount=show_amount),
        medical_block=build_medical_block(tr),
        route_steps=route_steps,
        mission_milestones=mission_milestones,
        synthetic_history=synthetic_history,
        timeline_rows=timeline_rows,
        timeline_truncated=timeline_truncated,
        messages=messages,
        messages_truncated=messages_truncated,
        route_legs_truncated=route_truncated,
        attachments=[],
        show_amount=show_amount,
        traceability={
            "public_id": getattr(tr, "public_id", None) or MISSING,
            "platform_label": "LIRIE",
        },
    )
    ctx.traceability["document_hash"] = compute_document_hash(ctx)
    ctx.traceability["generated_at_label"] = generated_at.strftime("%d.%m.%Y %H:%M")
    ctx.traceability["edition_date"] = generated_at.strftime("%d.%m.%Y")
    ctx.traceability["verify_url"] = "https://www.lirie.ch"
    ctx.traceability["verify_label"] = "Document généré via LIRIE"
    ctx.traceability["archive_reference"] = (
        f"LIRIE-{ctx.reference}-{ctx.traceability['document_hash']}"
    )
    ctx.completion_certificate = build_completion_certificate(
        reference=ctx.reference,
        status_label=status_label,
        patient_block=patient_block,
        institution_snapshot=institution_snapshot,
        carrier_block=carrier_block,
        mission_info=mission_info,
        document_hash=ctx.traceability["document_hash"],
        public_id=ctx.traceability["public_id"],
        generated_at=generated_at,
    )
    ctx.gps_proof = build_gps_proof(booking)
    return ctx

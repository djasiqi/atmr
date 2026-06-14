"""Service d'export des transports institution (PDF patient, PDF/CSV journalier).

Réutilise le pattern ReportLab `canvas` utilisé par `routes/bookings.py`
(`_build_client_bookings_pdf`). Les exports sont en lecture seule et destinés
aux rôles autorisés (admin, facturation, réception).

Trois exports sont fournis :
- PDF historique patient (sur une période)
- PDF journalier ("Transports du JJ.MM.AAAA") avec page de synthèse
- CSV journalier (mêmes lignes que le PDF, exploitable en tableur)
"""

from __future__ import annotations

import csv
import logging
import zipfile
from datetime import UTC, date, datetime
from io import BytesIO, StringIO
from typing import TYPE_CHECKING, Any

from models import InstitutionPatient, TransportRequest
from models.enums import CarrierSource, RequestStatus

if TYPE_CHECKING:
    from models.institution import Institution

logger = logging.getLogger(__name__)

# Seuil de saut de page (aligné sur routes/bookings.py)
_PDF_NEW_PAGE_Y_THRESHOLD = 80

# Libellés FR des statuts de demande / booking pour l'export
_REQUEST_STATUS_LABELS = {
    "DRAFT": "Brouillon",
    "SENT": "Envoyée",
    "ACCEPTED": "Acceptée",
    "CONVERTED": "Confirmée",
    "CANCELLED": "Annulée",
    "EXPIRED": "Expirée",
    "EXTERNAL_ASSIGNED": "Transporteur externe affecté",
    "EXTERNAL_DECLARED_COMPLETED": "Déclarée réalisée par l'institution",
}

_BOOKING_STATUS_LABELS = {
    "PENDING": "En attente",
    "ACCEPTED": "Accepté",
    "ASSIGNED": "Chauffeur assigné",
    "EN_ROUTE": "En route",
    "IN_PROGRESS": "En cours",
    "COMPLETED": "Terminé",
    "RETURN_COMPLETED": "Aller-retour terminé",
    "CANCELED": "Annulé",
}

_BILLING_INTENT_LABELS = {
    "patient": "Patient",
    "institution": "Institution",
    "curator": "Curateur",
    "spc": "SPC",
    "other": "Autre",
}

# Statuts considérés comme "transport effectué"
_COMPLETED_BOOKING_STATUSES = frozenset({"COMPLETED", "RETURN_COMPLETED"})


# ============================================================================
# Collecte des données
# ============================================================================


def collect_patient_transports(
    institution_id: int,
    patient_id: int,
    *,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[TransportRequest]:
    """Retourne les demandes de transport d'un patient sur une période."""
    query = TransportRequest.query.filter(
        TransportRequest.institution_id == institution_id,
        TransportRequest.patient_id == patient_id,
    )
    if date_from is not None:
        query = query.filter(TransportRequest.mission_date >= date_from.date())
    if date_to is not None:
        query = query.filter(TransportRequest.mission_date <= date_to.date())
    return query.order_by(
        TransportRequest.mission_date.asc(),
        TransportRequest.id.asc(),
    ).all()


def collect_daily_transports(
    institution_id: int,
    day_start: datetime,
    day_end: datetime,
) -> list[TransportRequest]:
    """Retourne toutes les demandes de transport d'une institution pour une journée."""
    return (
        TransportRequest.query.filter(
            TransportRequest.institution_id == institution_id,
            TransportRequest.mission_date >= day_start.date(),
            TransportRequest.mission_date < day_end.date(),
        )
        .order_by(TransportRequest.mission_date.asc(), TransportRequest.id.asc())
        .all()
    )


# ============================================================================
# Normalisation d'une ligne de transport
# ============================================================================


def _booking_status_str(booking: Any) -> str:
    raw = getattr(booking, "status", None)
    return str(getattr(raw, "value", raw) or "").upper()


def _request_status_str(req: TransportRequest) -> str:
    raw = getattr(req, "status", None)
    return str(getattr(raw, "value", raw) or "").upper()


def _driver_name(booking: Any) -> str | None:
    driver = getattr(booking, "driver", None)
    user = getattr(driver, "user", None) if driver else None
    if not user:
        return None
    name = f"{user.first_name or ''} {user.last_name or ''}".strip()
    return name or None


def _patient_label(req: TransportRequest) -> str:
    patient = getattr(req, "patient", None)
    if patient:
        return f"{patient.last_name} {patient.first_name}".strip()
    booking = getattr(req, "booking", None)
    if booking and getattr(booking, "customer_name", None):
        return str(booking.customer_name)
    return req.external_reference or f"#{req.id}"


def _status_label(req: TransportRequest, booking: Any) -> str:
    if booking is not None:
        status = _booking_status_str(booking)
        return _BOOKING_STATUS_LABELS.get(status, status or "—")
    status = _request_status_str(req)
    return RequestStatus.display_label(status) if status else "—"


def _carrier_source_label(req: TransportRequest) -> str:
    raw = getattr(req, "carrier_source", None) or CarrierSource.LIRIE.value
    source = str(getattr(raw, "value", raw) or CarrierSource.LIRIE.value)
    return CarrierSource.display_label(source)


def _carrier_name(req: TransportRequest, booking: Any) -> str:
    raw = getattr(req, "carrier_source", None) or CarrierSource.LIRIE.value
    source = str(getattr(raw, "value", raw) or CarrierSource.LIRIE.value)
    if source == CarrierSource.EXTERNAL.value:
        return getattr(req, "external_carrier_name", None) or "Transporteur externe"
    company = getattr(req, "accepted_by_company", None)
    if company is None and booking is not None:
        company = getattr(booking, "company", None)
    return getattr(company, "name", None) or "Non assignée"


def _is_completed(req: TransportRequest, booking: Any) -> bool:
    if booking is not None:
        return _booking_status_str(booking) in _COMPLETED_BOOKING_STATUSES
    raw = getattr(req, "carrier_source", None) or CarrierSource.LIRIE.value
    source = str(getattr(raw, "value", raw) or CarrierSource.LIRIE.value)
    return (
        source == CarrierSource.EXTERNAL.value
        and _request_status_str(req) == RequestStatus.EXTERNAL_DECLARED_COMPLETED.value
    )


def _fmt_dt(value: Any, fmt: str) -> str:
    if isinstance(value, datetime):
        return value.strftime(fmt)
    return "—"


def build_transport_row(req: TransportRequest) -> dict[str, Any]:
    """Construit une représentation plate d'une demande pour l'export."""
    from services.institutions.transport_request_display import (
        build_transport_request_display_blocks,
    )

    display = build_transport_request_display_blocks(req)
    scheduling = display.get("scheduling") or {}
    identity = display.get("identity") or {}
    booking = getattr(req, "booking", None)
    company = getattr(req, "accepted_by_company", None)
    scheduled = getattr(req, "scheduled_time", None)
    display_time = (scheduling.get("departure") or {}).get("display_time") or _fmt_dt(
        scheduled, "%H:%M"
    )
    is_completed = _is_completed(req, booking)
    carrier_name = _carrier_name(req, booking)
    return {
        "id": req.id,
        "patient_name": identity.get("primary_label") or _patient_label(req),
        "date": _fmt_dt(scheduled, "%d.%m.%Y"),
        "time": display_time,
        "schedule_summary": scheduling.get("summary") or "",
        "pickup_location": req.pickup_location or "—",
        "dropoff_location": req.dropoff_location or "—",
        "is_round_trip": bool(req.is_round_trip),
        "execution_mode_label": _carrier_source_label(req),
        "company_name": carrier_name,
        "external_carrier_reference": getattr(req, "external_carrier_reference", None) or "",
        "status_label": _status_label(req, booking),
        "billing_label": _BILLING_INTENT_LABELS.get(
            req.billing_intent, req.billing_intent or "—"
        ),
        "external_reference": req.external_reference or "",
        "is_completed": is_completed,
        "boarded_at": _fmt_dt(getattr(booking, "boarded_at", None), "%d.%m.%Y %H:%M"),
        "completed_at": (
            _fmt_dt(getattr(booking, "completed_at", None), "%d.%m.%Y %H:%M")
            if booking is not None
            else _fmt_dt(getattr(req, "executed_externally_at", None), "%d.%m.%Y %H:%M")
        ),
        "driver_name": _driver_name(booking) if booking else None,
        "distance_km": (
            round((getattr(booking, "distance_meters", None) or 0) / 1000.0, 1)
            if booking and getattr(booking, "distance_meters", None)
            else None
        ),
    }


# ============================================================================
# Statistiques de synthèse (page 1 PDF journalier)
# ============================================================================


def compute_daily_stats(requests: list[TransportRequest]) -> dict[str, Any]:
    """Calcule les statistiques de synthèse pour le PDF journalier."""
    total = len(requests)
    distinct_patients: set[int] = set()
    confirmed = 0
    cancelled = 0
    completed = 0
    completed_lirie = 0
    completed_external = 0
    round_trips = 0
    by_company: dict[str, int] = {}
    by_execution_mode: dict[str, int] = {"Transporteur LIRIE": 0, "Transporteur externe": 0}

    for req in requests:
        if req.patient_id is not None:
            distinct_patients.add(req.patient_id)
        if req.is_round_trip:
            round_trips += 1

        booking = getattr(req, "booking", None)
        execution_label = _carrier_source_label(req)
        by_execution_mode[execution_label] = by_execution_mode.get(execution_label, 0) + 1

        if booking is not None:
            status = _booking_status_str(booking)
            if status in _COMPLETED_BOOKING_STATUSES:
                completed += 1
                completed_lirie += 1
            if status == "CANCELED":
                cancelled += 1
        req_status = _request_status_str(req)
        if req_status == "CONVERTED":
            confirmed += 1
        elif req_status == "CANCELLED":
            cancelled += 1
        elif (
            req_status == RequestStatus.EXTERNAL_DECLARED_COMPLETED.value
            and execution_label == CarrierSource.display_label(CarrierSource.EXTERNAL.value)
        ):
            completed += 1
            completed_external += 1
        elif req_status == RequestStatus.EXTERNAL_ASSIGNED.value:
            confirmed += 1

        company_name = _carrier_name(req, booking)
        by_company[company_name] = by_company.get(company_name, 0) + 1

    return {
        "total_courses": total,
        "patients_transported": len(distinct_patients),
        "confirmed": confirmed,
        "completed": completed,
        "completed_lirie": completed_lirie,
        "completed_external": completed_external,
        "cancelled": cancelled,
        "round_trips": round_trips,
        "by_execution_mode": by_execution_mode,
        "by_company": dict(
            sorted(by_company.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }


# ============================================================================
# Helpers ReportLab
# ============================================================================


def _new_canvas() -> tuple[Any, Any, float, float]:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    return pdf, buffer, width, height


def _ensure_space(pdf: Any, y: float, height: float, *, font: str, size: int) -> float:
    if y < _PDF_NEW_PAGE_Y_THRESHOLD:
        pdf.showPage()
        pdf.setFont(font, size)
        return height - 40
    return y


def _draw_confirmation_block(
    pdf: Any, row: dict[str, Any], y: float, height: float
) -> float:
    """Dessine le bloc « Confirmation transport effectuée » pour une course."""
    y = _ensure_space(pdf, y, height, font="Helvetica", size=9)
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(48, y, "Confirmation transport effectuée")
    y -= 13
    pdf.setFont("Helvetica", 8)
    lines = [
        f"Patient : {row['patient_name']}",
        f"Trajet : {row['pickup_location']} -> {row['dropoff_location']}"
        + (" (aller-retour)" if row["is_round_trip"] else ""),
        f"Transporteur : {row['company_name']}"
        + (f" — Chauffeur : {row['driver_name']}" if row["driver_name"] else ""),
        f"Prise en charge : {row['boarded_at']}   Dépôt : {row['completed_at']}",
    ]
    if row["distance_km"] is not None:
        lines.append(f"Distance : {row['distance_km']} km")
    if row["external_reference"]:
        lines.append(f"Référence : {row['external_reference']}")
    for line in lines:
        y = _ensure_space(pdf, y, height, font="Helvetica", size=8)
        pdf.drawString(56, y, line[:120])
        y -= 11
    y -= 6
    return y


# ============================================================================
# PDF — Historique patient
# ============================================================================


def build_patient_pdf(
    institution: Institution,
    patient: InstitutionPatient,
    requests: list[TransportRequest],
    period_label: str,
) -> bytes:
    """Construit le PDF d'historique transports d'un patient."""
    pdf, buffer, _width, height = _new_canvas()
    y = height - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Historique des transports — Patient")
    y -= 20
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, y, f"Institution : {institution.name}")
    y -= 13
    pdf.drawString(40, y, f"Période : {period_label}")
    y -= 13
    generated = datetime.now(UTC).strftime("%d.%m.%Y %H:%M UTC")
    pdf.drawString(40, y, f"Généré le : {generated}")
    y -= 20

    # Bloc patient
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(40, y, f"{patient.last_name} {patient.first_name}")
    y -= 14
    pdf.setFont("Helvetica", 8)
    patient_lines = []
    if patient.dob:
        patient_lines.append(f"Né(e) le : {patient.dob.strftime('%d.%m.%Y')}")
    if patient.phone:
        patient_lines.append(f"Téléphone : {patient.phone}")
    if patient.insurance_name:
        patient_lines.append(f"Assurance : {patient.insurance_name}")
    if patient.external_reference:
        patient_lines.append(f"Réf. DPI : {patient.external_reference}")
    for line in patient_lines:
        pdf.drawString(48, y, line[:120])
        y -= 11
    y -= 8

    # Tableau des courses
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(40, y, "Date | Heure | Trajet | Transporteur | Statut | Facturation")
    y -= 13
    pdf.setFont("Helvetica", 8)

    if not requests:
        pdf.drawString(40, y, "Aucun transport sur la période sélectionnée.")
        y -= 12

    rows = [build_transport_row(req) for req in requests]
    for row in rows:
        y = _ensure_space(pdf, y, height, font="Helvetica", size=8)
        line = (
            f"{row['date']} | {row['time']} | "
            f"{row['pickup_location']} -> {row['dropoff_location']} | "
            f"{row['company_name']} | {row['status_label']} | {row['billing_label']}"
        )
        pdf.drawString(40, y, line[:130])
        y -= 12

    # Blocs de confirmation pour les transports effectués
    completed_rows = [r for r in rows if r["is_completed"]]
    if completed_rows:
        y -= 6
        y = _ensure_space(pdf, y, height, font="Helvetica-Bold", size=10)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(40, y, "Confirmations de transport")
        y -= 16
        for row in completed_rows:
            y = _draw_confirmation_block(pdf, row, y, height)

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# PDF — Journalier (avec page de synthèse)
# ============================================================================


def build_daily_pdf(
    institution: Institution,
    day_label: str,
    requests: list[TransportRequest],
) -> bytes:
    """Construit le PDF journalier avec une première page de synthèse."""
    pdf, buffer, _width, height = _new_canvas()
    y = height - 40

    # ── Page 1 : synthèse ──
    stats = compute_daily_stats(requests)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, f"Transports du {day_label}")
    y -= 22
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, y, f"Institution : {institution.name}")
    y -= 13
    generated = datetime.now(UTC).strftime("%d.%m.%Y %H:%M UTC")
    pdf.drawString(40, y, f"Généré le : {generated}")
    y -= 26

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Synthèse de la journée")
    y -= 18
    pdf.setFont("Helvetica", 10)
    summary_lines = [
        f"Patients transportés : {stats['patients_transported']}",
        f"Courses (demandes) : {stats['total_courses']}",
        f"Courses confirmées : {stats['confirmed']}",
        f"Transports effectués : {stats['completed']}",
        f"Aller-retours : {stats['round_trips']}",
        f"Annulées : {stats['cancelled']}",
    ]
    for line in summary_lines:
        pdf.drawString(48, y, line)
        y -= 15
    y -= 6

    if stats["by_company"]:
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, "Répartition par transporteur")
        y -= 16
        pdf.setFont("Helvetica", 9)
        for company_name, count in stats["by_company"].items():
            y = _ensure_space(pdf, y, height, font="Helvetica", size=9)
            pdf.drawString(48, y, f"{company_name} : {count}")
            y -= 13

    # ── Page 2+ : détail des courses ──
    pdf.showPage()
    y = height - 40
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(40, y, f"Détail des courses — {day_label}")
    y -= 20
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(
        40, y, "Heure | Patient | Trajet | Transporteur | Statut | Facturation"
    )
    y -= 13
    pdf.setFont("Helvetica", 8)

    if not requests:
        pdf.drawString(40, y, "Aucun transport pour cette journée.")
        y -= 12

    rows = [build_transport_row(req) for req in requests]
    for row in rows:
        y = _ensure_space(pdf, y, height, font="Helvetica", size=8)
        line = (
            f"{row['time']} | {row['patient_name']} | "
            f"{row['pickup_location']} -> {row['dropoff_location']} | "
            f"{row['company_name']} | {row['status_label']} | {row['billing_label']}"
        )
        pdf.drawString(40, y, line[:135])
        y -= 12

    # Blocs de confirmation pour les transports effectués
    completed_rows = [r for r in rows if r["is_completed"]]
    if completed_rows:
        y -= 6
        y = _ensure_space(pdf, y, height, font="Helvetica-Bold", size=10)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(40, y, "Confirmations de transport effectuées")
        y -= 16
        for row in completed_rows:
            y = _draw_confirmation_block(pdf, row, y, height)

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================================
# CSV — Journalier
# ============================================================================

_CSV_HEADERS = [
    "Date",
    "Heure",
    "Patient",
    "Référence",
    "Départ",
    "Arrivée",
    "Aller-retour",
    "Mode d'exécution",
    "Transporteur",
    "Référence externe",
    "Chauffeur",
    "Statut",
    "Facturation",
    "Prise en charge",
    "Dépôt",
    "Distance (km)",
]


def build_daily_csv(
    institution: Institution,
    day_label: str,
    requests: list[TransportRequest],
) -> bytes:
    """Construit le CSV journalier (UTF-8 BOM pour Excel)."""
    output = StringIO()
    writer = csv.writer(output, delimiter=";")
    writer.writerow([f"Transports du {day_label}", institution.name])
    writer.writerow(_CSV_HEADERS)

    for req in requests:
        row = build_transport_row(req)
        writer.writerow(
            [
                row["date"],
                row["time"],
                row["patient_name"],
                row["external_reference"],
                row["pickup_location"],
                row["dropoff_location"],
                "Oui" if row["is_round_trip"] else "Non",
                row["execution_mode_label"],
                row["company_name"],
                row["external_carrier_reference"],
                row["driver_name"] or "",
                row["status_label"],
                row["billing_label"],
                row["boarded_at"] if row["boarded_at"] != "—" else "",
                row["completed_at"] if row["completed_at"] != "—" else "",
                row["distance_km"] if row["distance_km"] is not None else "",
            ]
        )

    # BOM UTF-8 pour ouverture correcte des accents dans Excel
    return ("\ufeff" + output.getvalue()).encode("utf-8")


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """Retourne (début, fin exclusive) d'une journée en UTC."""
    from datetime import timedelta

    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


def build_daily_mission_reports_zip(
    institution: Institution,
    requests: list[TransportRequest],
) -> bytes:
    """Construit une archive ZIP : un rapport de mission (PDF audit) par demande."""
    from services.institutions.mission_report_context import (
        collect_mission_report_context,
        make_unique_mission_pdf_filenames,
    )
    from services.institutions.mission_report_pdf import build_mission_audit_report_pdf

    filenames = make_unique_mission_pdf_filenames(requests, variant="audit")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for tr in requests:
            ctx = collect_mission_report_context(
                tr,
                institution,
                variant="audit",
                show_amount=True,
            )
            pdf_bytes = build_mission_audit_report_pdf(ctx)
            archive.writestr(filenames[tr.id], pdf_bytes)
    buffer.seek(0)
    return buffer.getvalue()

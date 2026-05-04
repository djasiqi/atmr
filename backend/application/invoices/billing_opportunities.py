"""Agrégats read-only : payeurs / périodes avec transports encore à facturer (V2 dashboard)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy.orm import joinedload

from application.invoices.period_invoice_preview import build_period_invoice_preview
from ext import db
from models import Booking, Client, Company
from repositories.booking_repository import BookingRepository
from repositories.client_repository import ClientRepository

logger = logging.getLogger(__name__)

CALENDAR_MONTHS = 12


@dataclass(frozen=True, slots=True)
class PatientOpportunity:
    client_id: int
    display_name: str
    transports_count: int
    estimated_total: float
    currency: str


@dataclass(frozen=True, slots=True)
class ClinicOpportunity:
    clinic_company_id: int
    name: str
    transports_count: int
    estimated_total: float
    currency: str


@dataclass(frozen=True, slots=True)
class BillingOpportunitiesResult:
    period_year: int
    period_month: int
    patient_items: list[PatientOpportunity]
    clinic_items: list[ClinicOpportunity]
    # Repère batch (V3) : nombre de brouillons à créer si un « générer tout » existait
    total_draft_would_create: int


def _period_bounds(period_year: int, period_month: int) -> tuple[datetime, datetime]:
    start_date = datetime(period_year, period_month, 1)
    if period_month == CALENDAR_MONTHS:
        end_date = datetime(period_year + 1, 1, 1)
    else:
        end_date = datetime(period_year, period_month + 1, 1)
    return start_date, end_date


def _client_display_name(client: Client) -> str:
    user = getattr(client, "user", None)
    if user:
        fn = (getattr(user, "first_name", None) or "").strip()
        ln = (getattr(user, "last_name", None) or "").strip()
        if fn or ln:
            return f"{fn} {ln}".strip()
        un = (getattr(user, "username", None) or "").strip()
        if un:
            return un
    if client.institution_name:
        return str(client.institution_name)
    return f"Client #{client.id}"


def list_billing_opportunities(
    *,
    company_id: int,
    period_year: int,
    period_month: int,
) -> BillingOpportunitiesResult:
    """Liste les payeurs (patient direct + cliniques S2) avec au moins 1 transport éligible."""
    if not 1 <= period_month <= CALENDAR_MONTHS:
        raise ValueError("period_month invalide (1-12)")

    repo = BookingRepository()
    crepo = ClientRepository()
    start_date, end_date = _period_bounds(period_year, period_month)

    # --- Patient : clients distincts ayant au moins une course non facturée (piste large puis filtre repo) ---
    patient_items: list[PatientOpportunity] = []
    # JOIN Client : évite les client_id orphelins / autre entreprise (sinon period_invoice_preview lève).
    raw_client_ids = [
        r[0]
        for r in (
            db.session.query(Booking.client_id)
            .join(Client, Client.id == Booking.client_id)
            .filter(
                Booking.company_id == company_id,
                Client.company_id == company_id,
                Booking.invoice_line_id.is_(None),
                Booking.billed_to_type == "patient",
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )
            .distinct()
            .all()
        )
    ]
    for cid in raw_client_ids:
        if not crepo.find_model_by_id_and_company(int(cid), company_id):
            logger.warning(
                "billing_opportunities: client_id=%s ignoré (introuvable pour company_id=%s)",
                cid,
                company_id,
            )
            continue
        bookings = repo.find_models_unbilled_by_company_and_client(
            company_id, cid, period_year, period_month, billed_to_type="patient"
        )
        if not bookings:
            continue
        # Alignement montant : meme preview qu'avant generation
        prev = build_period_invoice_preview(
            company_id=company_id,
            period_year=period_year,
            period_month=period_month,
            client_id=cid,
            clinic_company_id=None,
        )
        client = (
            Client.query.options(joinedload(Client.user))
            .filter_by(id=cid)
            .first()
        )
        display = _client_display_name(client) if client else f"Client #{cid}"
        patient_items.append(
            PatientOpportunity(
                client_id=cid,
                display_name=display,
                transports_count=prev.transports_count,
                estimated_total=prev.estimated_total,
                currency=prev.currency,
            )
        )

    # --- S2 : cliniques (entreprises) avec transports billed_to_type=clinic (éligibles identiques à la preview) ---
    clinic_ids = [
        r[0]
        for r in (
            db.session.query(Booking.billed_to_company_id)
            .filter(
                Booking.company_id == company_id,
                Booking.billed_to_type == "clinic",
                Booking.invoice_line_id.is_(None),
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time < end_date,
            )
            .distinct()
            .all()
        )
    ]
    clinic_ids = [x for x in clinic_ids if x is not None]

    clinic_items: list[ClinicOpportunity] = []
    for ccid in sorted(set(clinic_ids)):
        prev = build_period_invoice_preview(
            company_id=company_id,
            period_year=period_year,
            period_month=period_month,
            client_id=None,
            clinic_company_id=ccid,
        )
        if prev.transports_count < 1:
            continue
        c = db.session.get(Company, ccid)
        name = c.name if c else f"Clinique #{ccid}"
        clinic_items.append(
            ClinicOpportunity(
                clinic_company_id=ccid,
                name=str(name or ""),
                transports_count=prev.transports_count,
                estimated_total=prev.estimated_total,
                currency=prev.currency,
            )
        )

    would_patient = sum(1 for p in patient_items if p.transports_count > 0)
    would_clinic = sum(1 for c in clinic_items if c.transports_count > 0)
    total_draft = would_patient + would_clinic

    return BillingOpportunitiesResult(
        period_year=period_year,
        period_month=period_month,
        patient_items=sorted(
            patient_items, key=lambda x: x.display_name.lower()
        ),
        clinic_items=sorted(clinic_items, key=lambda x: (x.name or "").lower()),
        total_draft_would_create=total_draft,
    )


def opportunities_to_dict(res: BillingOpportunitiesResult) -> dict[str, Any]:
    return {
        "period_year": res.period_year,
        "period_month": res.period_month,
        "patient_payers": [
            {
                "client_id": p.client_id,
                "display_name": p.display_name,
                "transports_count": p.transports_count,
                "estimated_total": p.estimated_total,
                "currency": p.currency,
            }
            for p in res.patient_items
        ],
        "clinic_payers": [
            {
                "clinic_company_id": c.clinic_company_id,
                "name": c.name,
                "transports_count": c.transports_count,
                "estimated_total": c.estimated_total,
                "currency": c.currency,
            }
            for c in res.clinic_items
        ],
        "total_draft_would_create": res.total_draft_would_create,
    }

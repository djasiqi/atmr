"""Monde G1 — 8 prestations cliniques 40 CHF + Marie DUPONT 40 CHF."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from models import Booking, Client, InstitutionPatient, TransportRequest, User
from models.enums import (
    BookingCreatedVia,
    BookingStatus,
    InstitutionBillingControlStatus,
    RequestStatus,
    UserRole,
)
from models.invoice import CompanyBillingSettings
from tests.e2e.helpers.billing_control_e2e import (
    make_clinic_payer_company,
    make_institution,
    make_transport_company,
    setup_clinic_billing_mapping,
)
from tests.e2e.helpers.institution_invoice_plan_lha import HUG, LHA

PERIOD_YEAR = 2026
PERIOD_MONTH = 8
LINE_HT = Decimal("40.00")


def _aug(day: int, hour: int = 10) -> datetime:
    return datetime(PERIOD_YEAR, PERIOD_MONTH, day, hour, 0, tzinfo=UTC)


def build_g1_clinic360_world(db) -> dict[str, Any]:
    institution = make_institution(db, name="Clinique les Hauts d'Anières")
    transport = make_transport_company(db)
    clinic = make_clinic_payer_company(db)
    clinic.name = "Clinique les Hauts d'Anières"
    db.session.flush()
    clinic_bp = setup_clinic_billing_mapping(
        db,
        transport_company=transport,
        clinic_company=clinic,
        institution=institution,
    )

    settings = CompanyBillingSettings()
    settings.company_id = transport.id
    settings.payment_terms_days = 30
    settings.vat_applicable = False
    settings.vat_rate = None
    db.session.add(settings)
    db.session.flush()

    suffix = uuid.uuid4().hex[:6]
    icu = User()
    icu.username = f"g1_{suffix}"
    icu.email = f"g1_{suffix}@e2e.ch"
    icu.role = UserRole.client
    icu.public_id = str(uuid.uuid4())
    icu.set_password("password123", force_change=False)
    db.session.add(icu)
    db.session.flush()

    clinic_client = Client()
    clinic_client.user_id = icu.id
    clinic_client.company_id = transport.id
    clinic_client.is_institution = True
    clinic_client.institution_name = clinic.name
    clinic_client.linked_institution_id = institution.id
    clinic_client.default_billed_to_company_id = clinic.id
    clinic_client.billing_address = institution.address
    db.session.add(clinic_client)
    db.session.flush()

    peers: list[Booking] = []
    for day in range(2, 10):
        booking = Booking()
        booking.company_id = transport.id
        booking.client_id = clinic_client.id
        booking.customer_name = f"Patient {day}"
        booking.pickup_location = LHA
        booking.dropoff_location = HUG
        booking.scheduled_time = _aug(day)
        booking.completed_at = _aug(day)
        booking.status = BookingStatus.COMPLETED.value
        booking.amount = LINE_HT
        booking.billed_to_type = "clinic"
        booking.billing_party_id = clinic_bp.id
        booking.billed_to_company_id = clinic.id
        booking.billing_origin = "OWN_PORTFOLIO"
        booking.created_via = BookingCreatedVia.DISPATCHER
        booking.institution_control_status = None
        db.session.add(booking)
        db.session.flush()
        peers.append(booking)

    dupont = InstitutionPatient()
    dupont.institution_id = institution.id
    dupont.first_name = "Marie"
    dupont.last_name = "DUPONT"
    dupont.address = LHA
    dupont.postal_code = "1247"
    dupont.city = "Anières"
    db.session.add(dupont)
    db.session.flush()

    marie = Booking()
    marie.company_id = transport.id
    marie.client_id = clinic_client.id
    marie.customer_name = "Marie DUPONT"
    marie.pickup_location = LHA
    marie.dropoff_location = HUG
    marie.scheduled_time = _aug(16, 9)
    marie.completed_at = _aug(16, 9)
    marie.status = BookingStatus.COMPLETED.value
    marie.amount = LINE_HT
    marie.billed_to_type = "clinic"
    marie.billing_party_id = clinic_bp.id
    marie.billed_to_company_id = clinic.id
    marie.institution_patient_id = dupont.id
    marie.billing_origin = "LIRIE_MARKETPLACE"
    marie.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    marie.institution_control_status = InstitutionBillingControlStatus.ANOMALY
    db.session.add(marie)
    db.session.flush()

    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.patient_id = dupont.id
    tr.external_reference = f"G1-{marie.id}"
    tr.pickup_location = marie.pickup_location
    tr.dropoff_location = marie.dropoff_location
    tr.scheduled_time = marie.scheduled_time
    tr.mission_date = marie.scheduled_time.date()
    tr.pickup_time_confirmed = True
    tr.status = RequestStatus.CONVERTED.value
    tr.billing_intent = "institution"
    tr.booking_id = marie.id
    db.session.add(tr)
    db.session.flush()

    return {
        "transport": transport,
        "clinic": clinic,
        "clinic_client": clinic_client,
        "peers": peers,
        "marie": marie,
        "all_clinic": [*peers, marie],
    }

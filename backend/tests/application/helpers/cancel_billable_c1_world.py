"""Monde C1 — éligibilité annulation facturable. Pas de montant C2, pas de PDF C4."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from models import Booking, Client, ClientStay, User
from models.enums import BookingCreatedVia, BookingStatus, UserRole
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
RIDE_HT = Decimal("90.00")
FEE_HT = Decimal("45.00")


def _aug(day: int, hour: int = 10) -> datetime:
    return datetime(PERIOD_YEAR, PERIOD_MONTH, day, hour, 0, tzinfo=UTC)


def build_c1_world(db) -> dict[str, Any]:
    institution = make_institution(db, name="Clinique C1")
    transport = make_transport_company(db)
    clinic = make_clinic_payer_company(db)
    clinic.name = "Clinique C1"
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
    icu.username = f"c1_{suffix}"
    icu.email = f"c1_{suffix}@e2e.ch"
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

    return {
        "transport": transport,
        "clinic": clinic,
        "clinic_client": clinic_client,
        "clinic_bp": clinic_bp,
        "institution": institution,
    }


def add_client_stay(
    db, *, client_id: int, clinic_id: int, when: datetime
) -> ClientStay:
    stay = ClientStay()
    stay.client_id = client_id
    stay.company_id = clinic_id
    stay.start_date = when
    stay.end_date = None
    stay.status = "active"
    db.session.add(stay)
    db.session.flush()
    return stay


def add_canceled_booking(
    db,
    world: dict[str, Any],
    *,
    billed_to_type: str,
    is_cancellation_billable: bool | None,
    cancellation_fee_amount: Decimal | None = FEE_HT,
    billing_override_reason: str | None = None,
    is_return: bool = False,
    parent_booking_id: int | None = None,
    day: int = 12,
) -> Booking:
    booking = Booking()
    booking.company_id = world["transport"].id
    booking.client_id = world["clinic_client"].id
    booking.customer_name = "Patient C1"
    booking.pickup_location = LHA
    booking.dropoff_location = HUG
    booking.scheduled_time = _aug(day)
    booking.completed_at = None
    booking.status = BookingStatus.CANCELED.value
    booking.amount = RIDE_HT
    booking.billed_to_type = billed_to_type
    booking.billing_party_id = world["clinic_bp"].id
    booking.billed_to_company_id = (
        world["clinic"].id if billed_to_type == "clinic" else None
    )
    booking.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    booking.is_return = is_return
    booking.parent_booking_id = parent_booking_id
    booking.is_cancellation_billable = is_cancellation_billable
    booking.cancellation_fee_amount = cancellation_fee_amount
    booking.billing_override_reason = billing_override_reason
    booking.invoice_line_id = None
    db.session.add(booking)
    db.session.flush()
    return booking

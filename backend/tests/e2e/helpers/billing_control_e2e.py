"""Fixtures E2E — contrôle facturation institution (INSTITUTION-07 BE1→BE12)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from flask_jwt_extended import create_access_token

from ext import db
from models import (
    BillingParty,
    Booking,
    Client,
    ClinicBillingPartyMapping,
    Company,
    Institution,
    InstitutionPatient,
    TransportRequest,
    User,
)
from models.enums import BillingPartyType, BookingStatus, RequestStatus, UserRole
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_institution_patient,
)

LIST_URL = "/api/v1/institutions/billing/control/bookings"


def institution_auth_headers(
    user: User,
    institution: Institution,
    role: str,
) -> dict[str, str]:
    from models.web_session import WebSession

    now = datetime.now(UTC)
    session = WebSession()
    session.id = str(uuid.uuid4())
    session.user_id = int(user.id)
    session.institution_id = institution.id
    session.created_at = now
    session.expires_at = now + timedelta(hours=8)
    session.last_interactive_activity_at = now
    db.session.add(session)
    db.session.flush()

    token = create_access_token(
        identity=str(user.public_id),
        additional_claims={
            "role": UserRole.INSTITUTION.value,
            "institution_id": institution.id,
            "institution_role": role,
            "sid": session.id,
            "aud": "atmr-api",
        },
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def make_institution(db, *, name: str | None = None) -> Institution:
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = name or f"Clinique E2E {uuid.uuid4().hex[:6]}"
    inst.institution_type = "clinic"
    inst.address = "Rue E2E 1, 1200 Genève"
    inst.billing_address = inst.address
    db.session.add(inst)
    db.session.flush()
    return inst


def make_institution_user(
    db,
    institution: Institution,
    *,
    role: str,
    prefix: str,
) -> User:
    user = User()
    user.username = f"{prefix}_{uuid.uuid4().hex[:6]}"
    user.email = f"{prefix}_{uuid.uuid4().hex[:6]}@e2e.ch"
    user.role = UserRole.INSTITUTION
    user.public_id = str(uuid.uuid4())
    user.institution_id = institution.id
    user.institution_role = role
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def make_transport_company(db) -> Company:
    cu = User()
    cu.username = f"co_{uuid.uuid4().hex[:6]}"
    cu.email = f"co_{uuid.uuid4().hex[:6]}@e2e.ch"
    cu.role = UserRole.company
    cu.public_id = str(uuid.uuid4())
    cu.set_password("password123", force_change=False)
    db.session.add(cu)
    db.session.flush()
    company = Company()
    company.name = f"Transport E2E {uuid.uuid4().hex[:4]}"
    company.address = "Rue Transport 1"
    company.contact_phone = "0210000000"
    company.contact_email = f"t_{uuid.uuid4().hex[:6]}@e2e.ch"
    company.user_id = cu.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    return company


def make_clinic_payer_company(db) -> Company:
    cu = User()
    cu.username = f"clu_{uuid.uuid4().hex[:6]}"
    cu.email = f"clu_{uuid.uuid4().hex[:6]}@e2e.ch"
    cu.role = UserRole.company
    cu.public_id = str(uuid.uuid4())
    cu.set_password("password123", force_change=False)
    db.session.add(cu)
    db.session.flush()
    clinic = Company()
    clinic.name = f"Clinique payeuse {uuid.uuid4().hex[:4]}"
    clinic.address = "Clinique addr"
    clinic.contact_phone = "0220000000"
    clinic.contact_email = f"c_{uuid.uuid4().hex[:6]}@e2e.ch"
    clinic.user_id = cu.id
    db.session.add(clinic)
    db.session.flush()
    return clinic


def setup_clinic_billing_mapping(
    db,
    *,
    transport_company: Company,
    clinic_company: Company,
    institution: Institution,
) -> BillingParty:
    bp = BillingParty()
    bp.company_id = transport_company.id
    bp.type = BillingPartyType.CLINIC
    bp.display_name = clinic_company.name
    bp.billing_address = clinic_company.address
    bp.external_ref = f"institution:{institution.id}"
    db.session.add(bp)
    db.session.flush()
    db.session.add(
        ClinicBillingPartyMapping(
            company_id=transport_company.id,
            clinic_company_id=clinic_company.id,
            billing_party_id=bp.id,
            is_active=True,
        )
    )
    db.session.flush()
    return bp


def make_eligible_control_booking(
    db,
    institution: Institution,
    *,
    transport_company: Company | None = None,
    scheduled: datetime | None = None,
    billed_to_type: str = "patient",
    status: str = BookingStatus.COMPLETED.value,
    billing_party_id: int | None = None,
    institution_patient_id: int | None = None,
    is_return: bool = False,
    parent_booking_id: int | None = None,
) -> tuple[Booking, TransportRequest, InstitutionPatient]:
    suffix = uuid.uuid4().hex[:8]
    when = scheduled or (datetime.now(UTC) + timedelta(days=2))
    company = transport_company or make_transport_company(db)

    icu = User()
    icu.username = f"icli_{suffix}"
    icu.email = f"icli_{suffix}@e2e.ch"
    icu.role = UserRole.client
    icu.public_id = str(uuid.uuid4())
    icu.set_password("password123", force_change=False)
    db.session.add(icu)
    db.session.flush()

    institution_client = Client()
    institution_client.user_id = icu.id
    institution_client.company_id = company.id
    institution_client.is_institution = True
    institution_client.institution_name = institution.name
    institution_client.billing_address = institution.address
    db.session.add(institution_client)
    db.session.flush()

    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = "Alice"
    patient.last_name = f"E2E{suffix[:4]}"
    patient.address = "Rue Patient"
    patient.postal_code = "1200"
    patient.city = "Genève"
    db.session.add(patient)
    db.session.flush()

    patient_bp = get_or_create_billing_party_for_institution_patient(
        company_id=company.id,
        institution_patient=patient,
    )

    booking = Booking()
    booking.company_id = company.id
    booking.client_id = institution_client.id
    booking.customer_name = f"{patient.first_name} {patient.last_name}"
    booking.pickup_location = "Domicile"
    booking.dropoff_location = "Clinique"
    booking.scheduled_time = when
    booking.completed_at = when
    booking.status = status
    booking.amount = Decimal("75.00")
    booking.billed_to_type = billed_to_type
    booking.billing_party_id = billing_party_id or int(patient_bp.id)
    booking.institution_patient_id = institution_patient_id or patient.id
    booking.is_return = is_return
    booking.parent_booking_id = parent_booking_id
    db.session.add(booking)
    db.session.flush()

    if not is_return:
        tr = TransportRequest()
        tr.public_id = str(uuid.uuid4())
        tr.institution_id = institution.id
        tr.patient_id = patient.id
        tr.external_reference = f"E2E-{suffix}"
        tr.pickup_location = booking.pickup_location
        tr.dropoff_location = booking.dropoff_location
        tr.scheduled_time = when
        tr.mission_date = when.date()
        tr.pickup_time_confirmed = True
        tr.status = RequestStatus.CONVERTED.value
        tr.billing_intent = "patient"
        tr.booking_id = booking.id
        db.session.add(tr)
        db.session.flush()
    else:
        tr = TransportRequest.query.filter_by(
            booking_id=parent_booking_id,
            institution_id=institution.id,
        ).first()
        assert tr is not None

    db.session.commit()
    return booking, tr, patient


def period_param(dt: datetime) -> str:
    return f"{dt.year}-{dt.month:02d}"


def assert_triplet_coherent(booking: Booking, payload: dict) -> None:
    payer = payload.get("payer") or payload
    assert payer.get("type") == booking.billed_to_type
    assert payer.get("billing_party_id") == getattr(booking, "billing_party_id", None)
    assert payer.get("billed_to_company_id") == getattr(
        booking, "billed_to_company_id", None
    )

"""Monde LHA / août 2026 — gate E2E institution_invoice_plan."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from models import (
    Booking,
    Client,
    InstitutionPatient,
    TransportRequest,
    User,
)
from models.enums import (
    BookingCreatedVia,
    BookingStatus,
    InstitutionBillingControlStatus,
    RequestStatus,
    UserRole,
)
from models.invoice import CompanyBillingSettings
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_institution_patient,
)
from tests.e2e.helpers.billing_control_e2e import (
    make_clinic_payer_company,
    make_institution,
    make_transport_company,
    setup_clinic_billing_mapping,
)

LHA = "Chemin des Courbes 9, 1247, Anières"
HUG = "Rue Gabrielle-Perret-Gentil 4, 1205, Genève"
PERIOD_YEAR = 2026
PERIOD_MONTH = 8


def _aug(day: int, hour: int = 10) -> datetime:
    return datetime(PERIOD_YEAR, PERIOD_MONTH, day, hour, 0, tzinfo=UTC)


def _patient(
    db,
    institution,
    *,
    first: str,
    last: str,
    company_id: int,
) -> tuple[InstitutionPatient, Any]:
    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = first
    patient.last_name = last
    patient.address = LHA
    patient.postal_code = "1247"
    patient.city = "Anières"
    db.session.add(patient)
    db.session.flush()
    bp = get_or_create_billing_party_for_institution_patient(
        company_id=company_id,
        institution_patient=patient,
    )
    return patient, bp


def _booking(
    db,
    *,
    company_id: int,
    client_id: int,
    patient: InstitutionPatient,
    when: datetime,
    billed_to_type: str,
    billing_party_id: int,
    billed_to_company_id: int | None,
    origin: str,
    created_via: BookingCreatedVia,
    control: InstitutionBillingControlStatus | None,
    pickup: str,
    dropoff: str,
    amount: Decimal = Decimal("40.00"),
    is_return: bool = False,
    parent_booking_id: int | None = None,
) -> Booking:
    booking = Booking()
    booking.company_id = company_id
    booking.client_id = client_id
    booking.customer_name = f"{patient.first_name} {patient.last_name}"
    booking.pickup_location = pickup
    booking.dropoff_location = dropoff
    booking.scheduled_time = when
    booking.completed_at = when
    booking.status = BookingStatus.COMPLETED.value
    booking.amount = amount
    booking.billed_to_type = billed_to_type
    booking.billing_party_id = billing_party_id
    booking.billed_to_company_id = billed_to_company_id
    booking.institution_patient_id = patient.id
    booking.is_return = is_return
    booking.parent_booking_id = parent_booking_id
    booking.billing_origin = origin
    booking.created_via = created_via
    booking.institution_control_status = control
    db.session.add(booking)
    db.session.flush()
    return booking


def _request(
    db,
    *,
    institution,
    patient: InstitutionPatient,
    booking: Booking,
    billing_intent: str,
) -> TransportRequest:
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.patient_id = patient.id
    tr.external_reference = f"LHA-{booking.id}"
    tr.pickup_location = booking.pickup_location
    tr.dropoff_location = booking.dropoff_location
    tr.scheduled_time = booking.scheduled_time
    tr.mission_date = booking.scheduled_time.date()
    tr.pickup_time_confirmed = True
    tr.status = RequestStatus.CONVERTED.value
    tr.billing_intent = billing_intent
    tr.booking_id = booking.id
    db.session.add(tr)
    db.session.flush()
    return tr


def build_lha_august_2026_world(db) -> dict[str, Any]:
    """12 prestations : portefeuille + Market, gates, payeurs, A/R, même jour non lié."""
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
    icu.username = f"lha_{suffix}"
    icu.email = f"lha_{suffix}@e2e.ch"
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

    alice, alice_bp = _patient(
        db, institution, first="Alice", last="MARTIN", company_id=transport.id
    )
    klein, klein_bp = _patient(
        db, institution, first="Arturo", last="KLEIN", company_id=transport.id
    )
    barbey, barbey_bp = _patient(
        db, institution, first="Jacques", last="BARBEY", company_id=transport.id
    )
    dupont, dupont_bp = _patient(
        db, institution, first="Marie", last="DUPONT", company_id=transport.id
    )
    cavadini, cavadini_bp = _patient(
        db,
        institution,
        first="Charlotte",
        last="CAVADINI",
        company_id=transport.id,
    )

    labels: dict[str, Booking] = {}
    portal = BookingCreatedVia.INSTITUTION_PORTAL
    dispatch = BookingCreatedVia.DISPATCHER
    validated = InstitutionBillingControlStatus.VALIDATED
    pending = InstitutionBillingControlStatus.PENDING_REVIEW
    disputed = InstitutionBillingControlStatus.ANOMALY

    def clinic_leg(**kwargs) -> Booking:
        return _booking(
            db,
            company_id=transport.id,
            client_id=clinic_client.id,
            billed_to_type="clinic",
            billed_to_company_id=clinic.id,
            billing_party_id=clinic_bp.id,
            **kwargs,
        )

    def patient_leg(*, patient, bp, **kwargs) -> Booking:
        return _booking(
            db,
            company_id=transport.id,
            client_id=clinic_client.id,
            patient=patient,
            billed_to_type="patient",
            billed_to_company_id=None,
            billing_party_id=int(bp.id),
            **kwargs,
        )

    labels["portfolio_clinic"] = clinic_leg(
        patient=alice,
        when=_aug(2),
        origin="OWN_PORTFOLIO",
        created_via=dispatch,
        control=None,
        pickup=LHA,
        dropoff=HUG,
    )

    labels["market_validated_clinic"] = clinic_leg(
        patient=alice,
        when=_aug(5),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=alice,
        booking=labels["market_validated_clinic"],
        billing_intent="institution",
    )

    labels["market_pending"] = clinic_leg(
        patient=alice,
        when=_aug(10),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=pending,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=alice,
        booking=labels["market_pending"],
        billing_intent="institution",
    )

    labels["market_disputed"] = clinic_leg(
        patient=alice,
        when=_aug(12),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=disputed,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=alice,
        booking=labels["market_disputed"],
        billing_intent="institution",
    )

    labels["market_validated_patient"] = patient_leg(
        patient=cavadini,
        bp=cavadini_bp,
        when=_aug(8),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=cavadini,
        booking=labels["market_validated_patient"],
        billing_intent="patient",
    )

    labels["ar_same_out"] = clinic_leg(
        patient=klein,
        when=_aug(15, 9),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=klein,
        booking=labels["ar_same_out"],
        billing_intent="institution",
    )
    labels["ar_same_ret"] = clinic_leg(
        patient=klein,
        when=_aug(15, 16),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=HUG,
        dropoff=LHA,
        is_return=True,
        parent_booking_id=labels["ar_same_out"].id,
    )

    labels["ar_split_out"] = clinic_leg(
        patient=dupont,
        when=_aug(16, 9),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=dupont,
        booking=labels["ar_split_out"],
        billing_intent="institution",
    )
    labels["ar_split_ret"] = patient_leg(
        patient=dupont,
        bp=dupont_bp,
        when=_aug(16, 16),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=HUG,
        dropoff=LHA,
        is_return=True,
        parent_booking_id=labels["ar_split_out"].id,
    )

    labels["same_day_a"] = clinic_leg(
        patient=barbey,
        when=_aug(18, 9),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=barbey,
        booking=labels["same_day_a"],
        billing_intent="institution",
    )
    labels["same_day_b"] = clinic_leg(
        patient=barbey,
        when=_aug(18, 16),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=HUG,
        dropoff=LHA,
    )
    _request(
        db,
        institution=institution,
        patient=barbey,
        booking=labels["same_day_b"],
        billing_intent="institution",
    )

    labels["financial_reopen"] = clinic_leg(
        patient=alice,
        when=_aug(20),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=alice,
        booking=labels["financial_reopen"],
        billing_intent="institution",
    )

    db.session.commit()
    for booking in labels.values():
        db.session.refresh(booking)

    return {
        "institution": institution,
        "transport": transport,
        "clinic": clinic,
        "clinic_client": clinic_client,
        "clinic_bp": clinic_bp,
        "bookings": labels,
        "patients": {
            "alice": alice,
            "klein": klein,
            "barbey": barbey,
            "dupont": dupont,
            "cavadini": cavadini,
        },
        "patient_bps": {
            "alice": alice_bp,
            "klein": klein_bp,
            "barbey": barbey_bp,
            "dupont": dupont_bp,
            "cavadini": cavadini_bp,
        },
        "period_year": PERIOD_YEAR,
        "period_month": PERIOD_MONTH,
    }

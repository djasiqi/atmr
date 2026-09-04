"""Seed local — transports LHA août 2026 + mot de passe institution.

Idempotent via billing_source_ref=lha_visual_aug2026.
Usage (Docker) :
  DISABLE_EVENTLET=1 python scripts/seed_lha_visual_aug2026.py
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from db import db  # noqa: E402
from models import (  # noqa: E402
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
from models.enums import (  # noqa: E402
    BillingPartyType,
    BookingCreatedVia,
    BookingStatus,
    InstitutionBillingControlStatus,
    RequestStatus,
)
from services.billing.billing_party_linker import (  # noqa: E402
    get_or_create_billing_party_for_institution_patient,
)

MARKER = "lha_visual_aug2026"
LHA_ADDR = "Chemin des Courbes 9, 1247, Anières"
HUG_ADDR = "Rue Gabrielle-Perret-Gentil 4, 1205, Genève"
INST_PASSWORD = "LhaAdmin1234"
AMOUNT = Decimal("40.00")


def _aug(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 8, day, hour, 0, tzinfo=UTC)


def _sep(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 9, day, hour, 0, tzinfo=UTC)


def _patient(institution: Institution, first: str, last: str) -> InstitutionPatient:
    existing = InstitutionPatient.query.filter_by(
        institution_id=institution.id, first_name=first, last_name=last
    ).first()
    if existing:
        return existing
    patient = InstitutionPatient()
    patient.public_id = str(uuid.uuid4())
    patient.institution_id = institution.id
    patient.first_name = first
    patient.last_name = last
    patient.address = LHA_ADDR
    patient.postal_code = "1247"
    patient.city = "Anières"
    db.session.add(patient)
    db.session.flush()
    return patient


def _clinic_party(company_id: int, institution: Institution) -> BillingParty:
    ref = f"institution:{institution.id}"
    existing = BillingParty.query.filter_by(
        company_id=company_id, external_ref=ref
    ).first()
    if existing:
        return existing
    bp = BillingParty()
    bp.company_id = company_id
    bp.type = BillingPartyType.CLINIC
    bp.display_name = institution.name
    bp.billing_address = institution.address or LHA_ADDR
    bp.contact_email = institution.contact_email
    bp.external_ref = ref
    db.session.add(bp)
    db.session.flush()
    mapping = ClinicBillingPartyMapping.query.filter_by(
        company_id=company_id, clinic_company_id=company_id
    ).first()
    if mapping is None:
        db.session.add(
            ClinicBillingPartyMapping(
                company_id=company_id,
                clinic_company_id=company_id,
                billing_party_id=bp.id,
                is_active=True,
            )
        )
        db.session.flush()
    return bp


def _booking(
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
    booking.amount = float(AMOUNT)
    booking.billed_to_type = billed_to_type
    booking.billing_party_id = billing_party_id
    booking.billed_to_company_id = billed_to_company_id
    booking.institution_patient_id = patient.id
    booking.is_return = is_return
    booking.parent_booking_id = parent_booking_id
    booking.billing_origin = origin
    booking.created_via = created_via
    booking.institution_control_status = control
    booking.billing_source_ref = MARKER
    db.session.add(booking)
    db.session.flush()
    return booking


def _request(
    *,
    institution: Institution,
    patient: InstitutionPatient,
    booking: Booking,
    billing_intent: str,
    created_by_user_id: int | None,
    created_by_display_name: str = "Admin LHA",
) -> TransportRequest:
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.patient_id = patient.id
    tr.created_by_user_id = created_by_user_id
    tr.created_by_display_name = created_by_display_name
    tr.external_reference = f"LHA-VISUAL-{booking.id}"
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


def seed() -> dict[str, int]:
    company = db.session.get(Company, 1)
    institution = db.session.get(Institution, 1)
    client = db.session.get(Client, 318)
    admin = User.query.filter_by(email="admin@lha.ch", institution_id=1).first()
    if company is None or institution is None or client is None:
        raise RuntimeError(
            "Monde LHA local introuvable (company 1 / institution 1 / client 318)."
        )

    if client.default_billed_to_company_id is None:
        client.default_billed_to_company_id = company.id

    already = Booking.query.filter_by(
        company_id=company.id, billing_source_ref=MARKER
    ).count()
    created = 0
    if already == 0:
        clinic_bp = _clinic_party(company.id, institution)
        alice = _patient(institution, "Alice", "MARTIN")
        klein = _patient(institution, "Arturo", "KLEIN")
        barbey = _patient(institution, "Jacques", "BARBEY")
        dupont = _patient(institution, "Marie", "DUPONT")
        cavadini = _patient(institution, "Charlotte", "CAVADINI")
        alice_bp = get_or_create_billing_party_for_institution_patient(
            company_id=company.id, institution_patient=alice
        )
        klein_bp = get_or_create_billing_party_for_institution_patient(
            company_id=company.id, institution_patient=klein
        )
        barbey_bp = get_or_create_billing_party_for_institution_patient(
            company_id=company.id, institution_patient=barbey
        )
        dupont_bp = get_or_create_billing_party_for_institution_patient(
            company_id=company.id, institution_patient=dupont
        )
        cavadini_bp = get_or_create_billing_party_for_institution_patient(
            company_id=company.id, institution_patient=cavadini
        )
        _ = (alice_bp, klein_bp, barbey_bp)

        portal = BookingCreatedVia.INSTITUTION_PORTAL
        dispatch = BookingCreatedVia.DISPATCHER
        validated = InstitutionBillingControlStatus.VALIDATED
        pending = InstitutionBillingControlStatus.PENDING_REVIEW
        disputed = InstitutionBillingControlStatus.ANOMALY
        admin_id = admin.id if admin else None

        def clinic_leg(**kwargs) -> Booking:
            return _booking(
                company_id=company.id,
                client_id=client.id,
                billed_to_type="clinic",
                billed_to_company_id=company.id,
                billing_party_id=clinic_bp.id,
                **kwargs,
            )

        def patient_leg(*, patient, bp, **kwargs) -> Booking:
            return _booking(
                company_id=company.id,
                client_id=client.id,
                patient=patient,
                billed_to_type="patient",
                billed_to_company_id=None,
                billing_party_id=int(bp.id),
                **kwargs,
            )

        portfolio = clinic_leg(
            patient=alice,
            when=_aug(2),
            origin="OWN_PORTFOLIO",
            created_via=dispatch,
            control=None,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        market_clinic = clinic_leg(
            patient=alice,
            when=_aug(5),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=alice,
            booking=market_clinic,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        pending_b = clinic_leg(
            patient=alice,
            when=_aug(10),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=pending,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=alice,
            booking=pending_b,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        disputed_b = clinic_leg(
            patient=alice,
            when=_aug(12),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=disputed,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=alice,
            booking=disputed_b,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        patient_b = patient_leg(
            patient=cavadini,
            bp=cavadini_bp,
            when=_aug(8),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=cavadini,
            booking=patient_b,
            billing_intent="patient",
            created_by_user_id=admin_id,
        )
        ar_out = clinic_leg(
            patient=klein,
            when=_aug(15, 9),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=klein,
            booking=ar_out,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        clinic_leg(
            patient=klein,
            when=_aug(15, 16),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=HUG_ADDR,
            dropoff=LHA_ADDR,
            is_return=True,
            parent_booking_id=ar_out.id,
        )
        split_out = clinic_leg(
            patient=dupont,
            when=_aug(16, 9),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=dupont,
            booking=split_out,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        patient_leg(
            patient=dupont,
            bp=dupont_bp,
            when=_aug(16, 16),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=HUG_ADDR,
            dropoff=LHA_ADDR,
            is_return=True,
            parent_booking_id=split_out.id,
        )
        same_a = clinic_leg(
            patient=barbey,
            when=_aug(18, 9),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=validated,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=barbey,
            booking=same_a,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        clinic_leg(
            patient=barbey,
            when=_aug(18, 16),
            origin="OWN_PORTFOLIO",
            created_via=dispatch,
            control=None,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        pending_sept = clinic_leg(
            patient=alice,
            when=_sep(3),
            origin="LIRIE_MARKETPLACE",
            created_via=portal,
            control=pending,
            pickup=LHA_ADDR,
            dropoff=HUG_ADDR,
        )
        _request(
            institution=institution,
            patient=alice,
            booking=pending_sept,
            billing_intent="institution",
            created_by_user_id=admin_id,
        )
        created = 13
        _ = portfolio

    if admin:
        admin.set_password(INST_PASSWORD, force_change=False)
        admin.account_status = "active"
        admin.token_version = int(admin.token_version or 0) + 1

    db.session.commit()
    total = Booking.query.filter_by(
        company_id=company.id, billing_source_ref=MARKER
    ).count()
    return {
        "created": created,
        "total_visual": total,
        "admin_id": int(admin.id) if admin else 0,
    }


def main() -> int:
    app = create_app()
    with app.app_context():
        result = seed()
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

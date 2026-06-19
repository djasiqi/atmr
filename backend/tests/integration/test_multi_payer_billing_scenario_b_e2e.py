# ruff: noqa: I001
"""STOP GATE P0 — facturation multi-payeurs par destination (scénario B).

EMS → HUG → Cabinet privé → Physiothérapie → EMS
Payeurs : Institution, Patient, Assurance (3 billing_party_id distincts).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import pytest

from application.institutions.accept_offer import AcceptOfferUseCase
from models import Booking, Company, Institution, InstitutionPatient, TransportRequest, TransportRequestLeg
from models.enums import RequestStatus
from services.billing.destination_billing_resolver import build_billing_summary
from shared.time_utils import now_local


@pytest.fixture
def ems_institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"EMS Multi-Payer B {uuid.uuid4().hex[:6]}"
    inst.institution_type = "ems"
    inst.address = "Rue EMS 2, 1200 Genève"
    inst.billing_address = "Rue EMS 2, 1200 Genève"
    db.session.add(inst)
    db.session.flush()
    return inst


@pytest.fixture
def institution_patient(db, ems_institution):
    patient = InstitutionPatient()
    patient.institution_id = ems_institution.id
    patient.first_name = "Paul"
    patient.last_name = "Durand"
    patient.address = "Chez Paul 8, 1200 Genève"
    patient.postal_code = "1200"
    patient.city = "Genève"
    db.session.add(patient)
    db.session.flush()
    return patient


def _future_depart(hour: int = 8) -> datetime:
    base = now_local() + timedelta(days=5)
    return base.replace(hour=hour, minute=0, second=0, microsecond=0)


def _build_scenario_b_request(
    db,
    *,
    institution: Institution,
    patient: InstitutionPatient,
) -> TransportRequest:
    route_group_id = str(uuid.uuid4())
    depart = _future_depart(8)
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.institution = institution
    tr.patient_id = patient.id
    tr.patient = patient
    tr.external_reference = f"MP-B-{uuid.uuid4().hex[:8]}"
    tr.pickup_location = "EMS Genève, Rue EMS 2"
    tr.dropoff_location = "HUG"
    tr.mission_date = depart.date()
    tr.scheduled_time = depart
    tr.pickup_time_confirmed = True
    tr.status = RequestStatus.SENT.value
    tr.multi_stop = True
    tr.return_to_institution = True
    tr.route_group_id = route_group_id
    tr.billing_intent = "institution"
    tr.billing_details = {
        "payer_name": "Assurance Maladie Test SA",
        "payer_address": "Av. Assurance 10, 1200 Genève",
        "insurance_number": "LAMAL-12345",
    }
    db.session.add(tr)
    db.session.flush()

    legs = [
        TransportRequestLeg(
            transport_request_id=tr.id,
            sequence_index=0,
            route_sequence_number=1,
            pickup_location="EMS Genève, Rue EMS 2",
            dropoff_location="HUG",
            dropoff_establishment="HUG",
            is_return_stop=False,
        ),
        TransportRequestLeg(
            transport_request_id=tr.id,
            sequence_index=1,
            route_sequence_number=2,
            pickup_location="HUG",
            dropoff_location="Cabinet privé, Genève",
            dropoff_establishment="Cabinet privé",
            destination_billing_override="patient",
            is_return_stop=False,
        ),
        TransportRequestLeg(
            transport_request_id=tr.id,
            sequence_index=2,
            route_sequence_number=3,
            pickup_location="Cabinet privé, Genève",
            dropoff_location="Physiothérapie Centre Nord",
            dropoff_establishment="Physiothérapie Centre Nord",
            destination_billing_override="insurance",
            is_return_stop=False,
        ),
        TransportRequestLeg(
            transport_request_id=tr.id,
            sequence_index=3,
            route_sequence_number=4,
            pickup_location="Physiothérapie Centre Nord",
            dropoff_location="EMS Genève, Rue EMS 2",
            is_return_stop=True,
        ),
    ]
    db.session.add_all(legs)
    db.session.flush()
    tr.legs = legs
    return tr


class TestMultiPayerBillingScenarioB:
    def test_scenario_b_three_distinct_billing_party_ids(
        self,
        db,
        requires_postgresql,
        ems_institution,
        institution_patient,
        test_company,
    ):
        if not test_company:
            pytest.skip("test_company required")

        clinic_co = Company(
            name=ems_institution.name,
            uid_ide="CHE-987.654.321",
            user_id=test_company.user_id,
        )
        db.session.add(clinic_co)
        db.session.flush()

        tr = _build_scenario_b_request(
            db,
            institution=ems_institution,
            patient=institution_patient,
        )

        uc = AcceptOfferUseCase()
        primary, _ = uc._create_bookings_from_legs(
            transport_request=tr,
            company_id=test_company.id,
            user_id=test_company.user_id,
        )
        db.session.flush()

        bookings = (
            Booking.query.filter_by(route_group_id=tr.route_group_id)
            .order_by(Booking.route_sequence_number.asc())
            .all()
        )
        assert len(bookings) == 4
        assert primary is not None

        by_seq = {b.route_sequence_number: b for b in bookings}
        institution_bp = by_seq[1].billing_party_id
        patient_bp = by_seq[2].billing_party_id
        insurance_bp = by_seq[3].billing_party_id
        return_bp = by_seq[4].billing_party_id

        assert institution_bp is not None
        assert patient_bp is not None
        assert insurance_bp is not None
        assert return_bp is not None

        assert patient_bp != insurance_bp
        assert patient_bp != institution_bp
        assert insurance_bp != institution_bp
        assert return_bp == institution_bp

        assert by_seq[2].billed_to_type == "patient"
        assert by_seq[3].billed_to_type == "insurance"

        summary = build_billing_summary(tr)
        assert summary["multi_payer"] is True
        assert summary["payer_count"] == 3
        assert len(summary["exceptions"]) == 2

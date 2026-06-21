"""Sérialisation RequestOffer pour l'API entreprise."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from models import (
    Company,
    Institution,
    InstitutionPatient,
    OfferMode,
    OfferStatus,
    RequestOffer,
    RequestStatus,
    TransportRequest,
    User,
    UserRole,
)


@pytest.fixture
def offer_with_patient(db):
    institution = Institution()
    institution.name = "Clinique Test"
    institution.public_id = str(uuid.uuid4())
    db.session.add(institution)
    db.session.flush()

    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = "Ramon"
    patient.last_name = "NYFFELER"
    patient.dob = date(1948, 3, 15)
    patient.external_reference = "PAT-001"
    db.session.add(patient)
    db.session.flush()

    user = User()
    user.email = f"company_{uuid.uuid4().hex[:8]}@test.com"
    user.username = user.email
    user.password = "test"
    user.role = UserRole.COMPANY.value
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = "Transport Test"
    company.user_id = user.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()

    scheduled = datetime.now(UTC) + timedelta(days=2)
    request = TransportRequest()
    request.institution_id = institution.id
    request.patient_id = patient.id
    request.external_reference = f"TEST-{uuid.uuid4().hex[:8]}"
    request.pickup_location = "Chemin des Courbes 9, 1247, Anières"
    request.dropoff_location = "Centre Médical, Genève"
    request.mission_date = scheduled.date()
    request.scheduled_time = scheduled
    request.status = RequestStatus.SENT.value
    db.session.add(request)
    db.session.flush()

    offer = RequestOffer(
        transport_request_id=request.id,
        company_id=company.id,
        mode=OfferMode.BROADCAST.value,
        status=OfferStatus.PENDING.value,
    )
    db.session.add(offer)
    db.session.flush()
    return offer, patient


def test_serialize_for_company_includes_patient_dob(db, offer_with_patient):
    offer, patient = offer_with_patient

    payload = offer.serialize_for_company()
    tr = payload["transport_request"]

    assert tr["patient_name"] == "Ramon NYFFELER"
    assert tr["patient"] == {
        "first_name": "Ramon",
        "last_name": "NYFFELER",
        "dob": patient.dob.isoformat(),
        "external_reference": "PAT-001",
    }


def test_serialize_for_company_patient_null_when_missing(db, offer_with_patient):
    offer, _patient = offer_with_patient
    offer.transport_request.patient_id = None
    offer.transport_request.patient = None
    db.session.flush()

    payload = offer.serialize_for_company()
    tr = payload["transport_request"]

    assert tr["patient_name"] is None
    assert tr["patient"] is None

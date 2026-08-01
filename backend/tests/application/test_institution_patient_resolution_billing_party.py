"""Résolution de l'identité patient depuis ``BillingParty.external_ref``."""

from __future__ import annotations

import uuid

import pytest

from application.invoices.institution_patient_resolution import (
    _patient_ids_from_billing_parties,
)
from models import BillingParty, BillingPartyType, Institution, InstitutionPatient


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"Clinique Résolution {uuid.uuid4().hex[:6]}"
    inst.institution_type = "clinic"
    inst.address = "Chemin des Courbes 9, 1247 Anières"
    inst.billing_address = "Chemin des Courbes 9, 1247 Anières"
    db.session.add(inst)
    db.session.flush()
    return inst


@pytest.fixture
def patient(db, institution):
    p = InstitutionPatient()
    p.institution_id = institution.id
    p.first_name = "Ali"
    p.last_name = "EL SAHBI"
    p.address = "Route d'Annecy 213A"
    p.postal_code = "1257"
    p.city = "Bardonnex"
    db.session.add(p)
    db.session.flush()
    return p


def _billing_party(db, *, external_ref: str | None, company_id: int = 1):
    bp = BillingParty()
    bp.company_id = company_id
    bp.type = BillingPartyType.PATIENT
    bp.display_name = "Ali EL SAHBI"
    bp.billing_address = "Route d'Annecy 213A\n1257 Bardonnex"
    bp.external_ref = external_ref
    bp.is_active = True
    db.session.add(bp)
    db.session.flush()
    return bp


def test_resolves_patient_from_external_ref(db, patient):
    bp = _billing_party(db, external_ref=f"patient:{patient.id}")

    assert _patient_ids_from_billing_parties({bp.id}) == {bp.id: patient.id}


def test_ignores_billing_party_without_patient_ref(db):
    bp = _billing_party(db, external_ref="institution:42")

    assert _patient_ids_from_billing_parties({bp.id}) == {}


def test_ignores_reference_to_missing_patient(db, patient):
    missing_id = patient.id + 10_000
    bp = _billing_party(db, external_ref=f"patient:{missing_id}")

    assert _patient_ids_from_billing_parties({bp.id}) == {}


def test_empty_input_does_not_query(db):
    assert _patient_ids_from_billing_parties(set()) == {}

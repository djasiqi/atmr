"""Applicabilité d'un mandat de démonstration. Aucune facture."""

from datetime import date

from domain.end_of_trip_settlement.mandate_applicability import (
    assess_mandate,
    decide_coverage,
    real_invoice_blocked,
)

_DAY = date(2026, 10, 30)


def _assess(**overrides):
    facts = {
        "mandate_company_id": 1,
        "booking_company_id": 1,
        "status": "demo_valid_simulated",
        "covered_tariff_versions": ("demo-2026.1",),
        "booking_tariff_version": None,
        "valid_from": None,
        "valid_until": None,
        "on_date": _DAY,
        "billed_to_type": "patient",
        "already_paid": False,
    }
    facts.update(overrides)
    return assess_mandate(**facts)


def test_institution_et_deja_paye_ne_concernent_pas_le_mandat():
    clinic = _assess(
        billed_to_type="clinic", status="demo_absent", mandate_company_id=None
    )
    paid = _assess(already_paid=True)
    assert clinic.kind == "not_concerned"
    assert paid.kind == "not_concerned"
    assert clinic.billing_authorized is False
    assert paid.billing_authorized is False


def test_brouillon_nest_pas_une_autorisation():
    decision = _assess(status="demo_draft")
    assert decision.kind == "prepared"
    assert decision.billing_authorized is False


def test_specimen_nest_pas_une_signature_juridique():
    decision = _assess(status="demo_specimen_signed")
    assert decision.kind == "specimen"
    assert decision.billing_authorized is False


def test_demonstration_applicable_sans_facture():
    decision = _assess()
    assert decision.kind == "applicable_demonstration"
    assert decision.label == "Mandat signé"
    assert decision.billing_authorized is False


def test_expiration_et_avenant_et_autre_entreprise():
    expired = _assess(valid_until=date(2026, 10, 1))
    amendment = _assess(booking_tariff_version="9")
    foreign = _assess(booking_company_id=2)
    assert expired.kind == "expired"
    assert amendment.kind == "amendment_required"
    assert amendment.amendment_required is True
    assert foreign.kind == "other_company"
    assert amendment.label == "Avenant requis — version tarifaire non couverte."
    assert expired.billing_authorized is False


def test_aucune_situation_nautorise_une_facture_reelle():
    kinds = (
        "absent",
        "prepared",
        "specimen",
        "applicable_demonstration",
        "expired",
        "amendment_required",
        "other_company",
        "not_concerned",
        "incomplete",
    )
    reasons = {
        real_invoice_blocked(
            kind=kind, legally_effective=False, real_signature_collected=False
        )
        for kind in kinds
    }
    assert "convention_inexistante" in reasons
    assert "convention_preparee" in reasons
    assert "mandat_demonstration" in reasons
    assert "convention_expiree" in reasons
    assert "avenant_non_signe" in reasons
    assert "autre_entreprise" in reasons
    flagged = real_invoice_blocked(
        kind="applicable_demonstration",
        legally_effective=True,
        real_signature_collected=True,
    )
    assert flagged == "preuve_juridique_absente"


def test_couverture_technique_sans_facture():
    covered = decide_coverage(
        mandate_client_id=7,
        booking_client_id=7,
        mandate_company_id=3,
        booking_company_id=3,
        active=True,
        covered_tariffs=("demo-2026.1",),
        booking_tariff="demo-2026.1",
        covered_services=("transport",),
        booking_service="transport",
    )
    tariff = decide_coverage(
        mandate_client_id=7,
        booking_client_id=7,
        mandate_company_id=3,
        booking_company_id=3,
        active=True,
        covered_tariffs=("demo-2026.1",),
        booking_tariff="demo-2026.2",
        covered_services=("transport",),
        booking_service="transport",
    )
    service = decide_coverage(
        mandate_client_id=7,
        booking_client_id=7,
        mandate_company_id=3,
        booking_company_id=3,
        active=True,
        covered_tariffs=("demo-2026.1",),
        booking_tariff="demo-2026.1",
        covered_services=("transport",),
        booking_service="brancard",
    )
    foreign = decide_coverage(
        mandate_client_id=7,
        booking_client_id=7,
        mandate_company_id=3,
        booking_company_id=9,
        active=True,
        covered_tariffs=("demo-2026.1",),
        booking_tariff="demo-2026.1",
        covered_services=("transport",),
        booking_service="transport",
    )
    other = decide_coverage(
        mandate_client_id=7,
        booking_client_id=8,
        mandate_company_id=3,
        booking_company_id=3,
        active=True,
        covered_tariffs=("demo-2026.1",),
        booking_tariff="demo-2026.1",
        covered_services=(),
        booking_service=None,
    )
    missing = decide_coverage(
        mandate_client_id=None,
        booking_client_id=7,
        mandate_company_id=None,
        booking_company_id=3,
        active=False,
        covered_tariffs=(),
        booking_tariff=None,
        covered_services=(),
        booking_service=None,
    )
    inactive = decide_coverage(
        mandate_client_id=7,
        booking_client_id=7,
        mandate_company_id=3,
        booking_company_id=3,
        active=False,
        covered_tariffs=("demo-2026.1",),
        booking_tariff="demo-2026.1",
        covered_services=("transport",),
        booking_service="transport",
    )
    assert covered.code == "applicable"
    assert covered.billing_authorized is False
    assert covered.legally_effective is False
    assert tariff.code == "avenant_requis"
    assert tariff.motif == "tarif"
    assert service.code == "avenant_requis"
    assert service.motif == "prestation"
    assert foreign.code == "autre_entreprise"
    assert other.code == "autre_patient"
    assert missing.code == "mandat_absent"
    assert inactive.code == "mandat_absent"

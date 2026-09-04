"""Extensions LHA — patients supplémentaires pour le gate batch (sans modifier LHA)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from models.enums import BookingCreatedVia, InstitutionBillingControlStatus
from tests.e2e.helpers.institution_invoice_plan_lha import (
    HUG,
    LHA,
    _aug,
    _booking,
    _patient,
    _request,
)


def extend_lha_world_for_patient_batch(db, world: dict[str, Any]) -> dict[str, Any]:
    """Ajoute des débiteurs patients pour BE2 / BE4 / BE10–BE13."""
    institution = world["institution"]
    transport = world["transport"]
    clinic_client = world["clinic_client"]
    portal = BookingCreatedVia.INSTITUTION_PORTAL
    validated = InstitutionBillingControlStatus.VALIDATED
    pending = InstitutionBillingControlStatus.PENDING_REVIEW
    disputed = InstitutionBillingControlStatus.ANOMALY

    moretti, moretti_bp = _patient(
        db, institution, first="Luca", last="MORETTI", company_id=transport.id
    )
    rivet, rivet_bp = _patient(
        db, institution, first="Eva", last="RIVET", company_id=transport.id
    )
    rossi, rossi_bp = _patient(
        db, institution, first="Gina", last="ROSSI", company_id=transport.id
    )
    bianchi, bianchi_bp = _patient(
        db, institution, first="Pia", last="BIANCHI", company_id=transport.id
    )
    verdi, verdi_bp = _patient(
        db, institution, first="Nino", last="VERDI", company_id=transport.id
    )
    faure, faure_bp = _patient(
        db, institution, first="Lise", last="FAURE", company_id=transport.id
    )

    def patient_leg(*, patient, bp, **kwargs):
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

    labels = world["bookings"]
    labels["moretti_a"] = patient_leg(
        patient=moretti,
        bp=moretti_bp,
        when=_aug(6, 9),
        origin="OWN_PORTFOLIO",
        created_via=BookingCreatedVia.DISPATCHER,
        control=None,
        pickup=LHA,
        dropoff=HUG,
        amount=Decimal("40.00"),
    )
    labels["moretti_b"] = patient_leg(
        patient=moretti,
        bp=moretti_bp,
        when=_aug(7, 9),
        origin="OWN_PORTFOLIO",
        created_via=BookingCreatedVia.DISPATCHER,
        control=None,
        pickup=HUG,
        dropoff=LHA,
        amount=Decimal("40.00"),
    )

    labels["rivet_out"] = patient_leg(
        patient=rivet,
        bp=rivet_bp,
        when=_aug(21, 9),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=rivet,
        booking=labels["rivet_out"],
        billing_intent="patient",
    )
    labels["rivet_ret"] = patient_leg(
        patient=rivet,
        bp=rivet_bp,
        when=_aug(21, 16),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=HUG,
        dropoff=LHA,
        is_return=True,
        parent_booking_id=labels["rivet_out"].id,
    )

    labels["rossi_pending"] = patient_leg(
        patient=rossi,
        bp=rossi_bp,
        when=_aug(22),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=pending,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=rossi,
        booking=labels["rossi_pending"],
        billing_intent="patient",
    )

    labels["bianchi_disputed"] = patient_leg(
        patient=bianchi,
        bp=bianchi_bp,
        when=_aug(23),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=disputed,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=bianchi,
        booking=labels["bianchi_disputed"],
        billing_intent="patient",
    )

    labels["verdi_pending"] = patient_leg(
        patient=verdi,
        bp=verdi_bp,
        when=_aug(24),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=pending,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=verdi,
        booking=labels["verdi_pending"],
        billing_intent="patient",
    )

    labels["faure_reopen"] = patient_leg(
        patient=faure,
        bp=faure_bp,
        when=_aug(25),
        origin="LIRIE_MARKETPLACE",
        created_via=portal,
        control=validated,
        pickup=LHA,
        dropoff=HUG,
    )
    _request(
        db,
        institution=institution,
        patient=faure,
        booking=labels["faure_reopen"],
        billing_intent="patient",
    )

    db.session.commit()
    for booking in labels.values():
        db.session.refresh(booking)

    world["patients"].update(
        {
            "moretti": moretti,
            "rivet": rivet,
            "rossi": rossi,
            "bianchi": bianchi,
            "verdi": verdi,
            "faure": faure,
        }
    )
    world["patient_bps"].update(
        {
            "moretti": moretti_bp,
            "rivet": rivet_bp,
            "rossi": rossi_bp,
            "bianchi": bianchi_bp,
            "verdi": verdi_bp,
            "faure": faure_bp,
        }
    )
    return world

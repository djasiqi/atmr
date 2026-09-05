"""G1 — matrice financière dérivée de G2. Pas de PDF, pas de QR, pas d'UX."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.booking_dispute import g1_financials as g1
from application.invoices.booking_dispute import service as svc
from models.enums import InstitutionBillingControlStatus

BASE_CLINIC = 320.0
MARIE_HT = 40.0
TOTAL_OPEN = 360.0
TOTAL_BLOCKED = 320.0


def _peer(i: int):
    return SimpleNamespace(
        id=100 + i,
        amount=40.0,
        billed_to_type="clinic",
        billing_party_id=10,
        invoice_billing_status=None,
        billing_origin="OWN_PORTFOLIO",
        created_via="dispatcher",
        institution_control_status=None,
        invoice_line_id=None,
        status=SimpleNamespace(value="completed"),
        _resolve_source_transport_request=lambda: None,
    )


def _marie(**kwargs):
    defaults = {
        "id": 45705,
        "company_id": 1,
        "amount": 40.0,
        "billed_to_type": "clinic",
        "billing_party_id": 10,
        "invoice_billing_status": None,
        "billing_origin": "LIRIE_MARKETPLACE",
        "created_via": "institution_portal",
        "institution_control_status": InstitutionBillingControlStatus.ANOMALY,
        "institution_control_anomaly_reason": "TRANSPORT_DISPUTED",
        "customer_name": "Marie DUPONT",
        "institution_patient": None,
        "scheduled_time": None,
        "completed_at": None,
        "driver_id": 3,
        "status": SimpleNamespace(value="completed"),
        "pickup_lat": 46.2,
        "dropoff_lat": 46.2,
        "invoice_line_id": None,
        "_resolve_source_transport_request": lambda: None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _dispute(**kwargs):
    defaults = {
        "id": 1,
        "booking_id": 45705,
        "status": "disputed",
        "carrier_stance": None,
        "proposed_amount_ht": None,
        "proposed_payer_type": None,
        "proposed_correction_note": None,
        "evidence": [],
        "events": [],
        "frozen_amount_ht": 40,
        "frozen_payer_type": "clinic",
        "carrier_exclusion_reason": None,
        "carrier_note": None,
        "carrier_responded_at": None,
        "carrier_responded_by_user_id": None,
        "submitted_at": None,
        "resolved_at": None,
        "resolved_by_user_id": None,
        "resolver_role": None,
        "resolution_note": None,
        "institution_reason_code": "TRANSPORT_DISPUTED",
        "institution_reason_text": "TRANSPORT_DISPUTED",
        "opened_at": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _patch(monkeypatch, dispute):
    monkeypatch.setattr(svc, "ensure_open_dispute", lambda *_a, **_k: dispute)
    monkeypatch.setattr(svc, "get_open_dispute", lambda *_a, **_k: dispute)
    monkeypatch.setattr(svc, "bump_edit_version", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_append_event", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_record_billing_event", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_ensure_system_snapshot", lambda *_a, **_k: None)

    class _Evidence:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            dispute.evidence = [*list(dispute.evidence), self]

    monkeypatch.setattr(svc, "BookingDisputeEvidence", _Evidence)
    monkeypatch.setattr(svc.db.session, "add", lambda *_a, **_k: None)


def _world(marie):
    return [*[_peer(i) for i in range(8)], marie]


def _assert_surface(bookings, *, total: float, marie_in: bool, marie_id: int = 45705):
    surface = g1.institution_surface(bookings)
    preview = g1.preview_institution_total(bookings)
    assert surface["institution_total"] == total
    assert preview == total
    assert preview == surface["institution_total"]
    if marie_in:
        assert marie_id in surface["eligible_lines"]
        assert marie_id not in surface["excluded_lines"]
    else:
        assert marie_id in surface["excluded_lines"]
        assert marie_id not in surface["eligible_lines"]
    return surface


def _assert_line(booking, dispute, *, billable: bool, payer: str, amount: float):
    row = g1.line_financials(booking, dispute)
    assert row["is_billable_to_institution"] is billable
    assert row["effective_payer"] == payer
    assert row["effective_amount"] == amount
    return row


def test_g1_aucune_contestation_360():
    marie = _marie(institution_control_status=InstitutionBillingControlStatus.VALIDATED)
    bookings = _world(marie)
    _assert_line(marie, None, billable=True, payer="clinic", amount=MARIE_HT)
    _assert_surface(bookings, total=TOTAL_OPEN, marie_in=True)


def test_g1_disputed_320():
    marie = _marie()
    dispute = _dispute()
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)


def test_g1_resolved_institution_320(monkeypatch):
    marie = _marie()
    dispute = _dispute()
    _patch(monkeypatch, dispute)
    result = svc.carrier_respond(
        marie,
        stance="institution_right",
        exclusion_reason="created_by_error",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert result.ok is True
    assert marie.billed_to_type == "clinic"
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)


def test_g1_mission_done_until_third_party_stays_320(monkeypatch):
    marie = _marie()
    dispute = _dispute()
    _patch(monkeypatch, dispute)

    svc.carrier_respond(
        marie, stance="mission_done", actor_user_id=2, actor_role="COMPANY"
    )
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    denied = svc.decide_dispute(
        marie,
        decision="accept_carrier",
        note=None,
        actor_user_id=9,
        actor_role="COMPANY",
    )
    assert denied.status_code == 403
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    svc.add_carrier_evidence(
        marie, kind="signed_transport_sheet", actor_user_id=2, actor_role="COMPANY"
    )
    svc.submit_dispute_for_validation(marie, actor_user_id=2, actor_role="COMPANY")
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    svc.decide_dispute(
        marie,
        decision="reject_evidence",
        note="illisible",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    svc.submit_dispute_for_validation(marie, actor_user_id=2, actor_role="COMPANY")
    svc.decide_dispute(
        marie,
        decision="accept_carrier",
        note="ok",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    _assert_line(marie, dispute, billable=True, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_OPEN, marie_in=True)


def test_g1_correction_never_consumes_proposed_until_validated(monkeypatch):
    marie = _marie()
    dispute = _dispute()
    _patch(monkeypatch, dispute)

    partner = svc.carrier_respond(
        marie,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="partner",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert partner.status_code == 400
    assert marie.amount == MARIE_HT
    assert marie.billed_to_type == "clinic"
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    svc.carrier_respond(
        marie,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="clinic",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    row = _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    assert row["proposed_amount_ht"] == 35.0
    assert row["effective_amount"] == MARIE_HT
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    svc.add_carrier_evidence(
        marie, kind="signed_transport_sheet", actor_user_id=2, actor_role="COMPANY"
    )
    svc.submit_dispute_for_validation(marie, actor_user_id=2, actor_role="COMPANY")
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    ambiguous = svc.carrier_respond(
        marie,
        stance="institution_right",
        exclusion_reason="other",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert ambiguous.status_code == 409
    _assert_line(marie, dispute, billable=False, payer="clinic", amount=MARIE_HT)
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

    svc.decide_dispute(
        marie,
        decision="accept_carrier",
        note="35 ok",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    _assert_line(marie, dispute, billable=True, payer="clinic", amount=35.0)
    _assert_surface(_world(marie), total=BASE_CLINIC + 35.0, marie_in=True)


def test_g1_correction_patient_validated_stays_320_institution(monkeypatch):
    marie = _marie()
    dispute = _dispute()
    _patch(monkeypatch, dispute)
    svc.carrier_respond(
        marie,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="patient",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    svc.add_carrier_evidence(
        marie, kind="institution_written", actor_user_id=2, actor_role="COMPANY"
    )
    svc.submit_dispute_for_validation(marie, actor_user_id=2, actor_role="COMPANY")
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)
    svc.decide_dispute(
        marie,
        decision="accept_carrier",
        note="à charge patient",
        actor_user_id=8,
        actor_role="admin",
    )
    row = g1.line_financials(marie, dispute)
    assert row["effective_payer"] == "patient"
    assert row["effective_amount"] == 35.0
    assert row["is_billable_to_institution"] is False
    _assert_surface(_world(marie), total=TOTAL_BLOCKED, marie_in=False)

"""G2 — machine d'état des 3 branches. Pas de PDF, pas de QR, pas de matrice G1."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.booking_dispute import machine as machine_mod
from application.invoices.booking_dispute import service as svc
from models.enums import InstitutionBillingControlStatus


def _booking(**kwargs):
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
        "carrier_exclusion_reason": None,
        "carrier_note": None,
        "carrier_responded_at": None,
        "carrier_responded_by_user_id": None,
        "proposed_amount_ht": None,
        "proposed_payer_type": None,
        "proposed_correction_note": None,
        "submitted_at": None,
        "resolved_at": None,
        "resolved_by_user_id": None,
        "resolver_role": None,
        "resolution_note": None,
        "evidence": [],
        "events": [],
        "frozen_amount_ht": 40,
        "frozen_payer_type": "clinic",
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


def _snap(booking, dispute):
    return machine_mod.snapshot(booking, dispute)


def _assert_blocked(state):
    assert state["open"] is True
    assert state["terminal"] is False
    assert state["clinic_line_in_invoice"] is False
    assert state["bucket"] == "disputed_blocked"
    assert state["carrier_can_close"] is False


def _assert_company_cannot_decide(booking, monkeypatch, dispute):
    result = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note=None,
        actor_user_id=9,
        actor_role="COMPANY",
    )
    assert result.ok is False
    assert result.status_code == 403
    _patch(monkeypatch, dispute)


def test_g2_branch_a_institution_right(monkeypatch):
    booking = _booking()
    dispute = _dispute()
    _patch(monkeypatch, dispute)

    initial = _snap(booking, dispute)
    _assert_blocked(initial)
    assert initial["status"] == "disputed"
    assert initial["who_may_act"] == ("COMPANY",)
    assert initial["payer"] == "clinic"

    result = svc.carrier_respond(
        booking,
        stance="institution_right",
        exclusion_reason="created_by_error",
        note="course créée par erreur",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert result.ok is True

    final = _snap(booking, dispute)
    assert final["status"] == "resolved_institution"
    assert final["stance"] == "institution_right"
    assert final["open"] is False
    assert final["terminal"] is True
    assert final["who_may_act"] == ()
    assert final["clinic_line_in_invoice"] is False
    assert final["bucket"] == "other_excluded"
    assert final["exclusion_reason"] == "resolved_institution_not_billable"
    assert final["invoice_billing_status"] == "not_billable"
    assert final["payer"] == "clinic"
    assert booking.billed_to_type == "clinic"
    assert not hasattr(booking, "deleted_at")

    blocked = svc.carrier_respond(
        booking,
        stance="mission_done",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert blocked.ok is False
    assert blocked.status_code == 409


def test_g2_branch_b_mission_done_third_party_only(monkeypatch):
    booking = _booking()
    dispute = _dispute()
    _patch(monkeypatch, dispute)

    initial = _snap(booking, dispute)
    _assert_blocked(initial)
    assert initial["who_may_act"] == ("COMPANY",)

    responded = svc.carrier_respond(
        booking,
        stance="mission_done",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert responded.ok is True
    mid = _snap(booking, dispute)
    assert mid["status"] == "awaiting_carrier_response"
    assert mid["stance"] == "mission_done"
    assert mid["who_may_act"] == ("COMPANY",)
    _assert_blocked(mid)
    assert mid["amount_ht"] == 40.0
    _assert_company_cannot_decide(booking, monkeypatch, dispute)

    early_decide = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note=None,
        actor_user_id=8,
        actor_role="institution_admin",
    )
    assert early_decide.ok is False
    assert early_decide.status_code == 409
    _assert_blocked(_snap(booking, dispute))

    evidence = svc.add_carrier_evidence(
        booking,
        kind="signed_transport_sheet",
        note="bon signé",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert evidence.ok is True
    _assert_blocked(_snap(booking, dispute))

    submitted = svc.submit_dispute_for_validation(
        booking, actor_user_id=2, actor_role="COMPANY"
    )
    assert submitted.ok is True
    pending = _snap(booking, dispute)
    assert pending["status"] == "evidence_submitted"
    assert pending["who_may_act"] == ("institution", "admin")
    _assert_blocked(pending)

    carrier_close = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note=None,
        actor_user_id=9,
        actor_role="COMPANY",
    )
    assert carrier_close.ok is False
    assert carrier_close.status_code == 403
    _assert_blocked(_snap(booking, dispute))

    stance_after_submit = svc.carrier_respond(
        booking,
        stance="institution_right",
        exclusion_reason="other",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert stance_after_submit.ok is False
    assert stance_after_submit.status_code == 409

    rejected = svc.decide_dispute(
        booking,
        decision="reject_evidence",
        note="illisible",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    assert rejected.ok is True
    after_reject = _snap(booking, dispute)
    assert after_reject["status"] == "awaiting_carrier_response"
    assert after_reject["who_may_act"] == ("COMPANY",)
    _assert_blocked(after_reject)

    resubmit = svc.submit_dispute_for_validation(
        booking, actor_user_id=2, actor_role="COMPANY"
    )
    assert resubmit.ok is True
    _assert_blocked(_snap(booking, dispute))

    accepted = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note="ok",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    assert accepted.ok is True
    final = _snap(booking, dispute)
    assert final["status"] == "resolved_carrier"
    assert final["open"] is False
    assert final["terminal"] is True
    assert final["who_may_act"] == ()
    assert final["clinic_line_in_invoice"] is True
    assert final["bucket"] == "clinic_billable"
    assert final["payer"] == "clinic"
    assert final["amount_ht"] == 40.0


def test_g2_branch_c_correction_amount_payer_only(monkeypatch):
    booking = _booking()
    dispute = _dispute()
    _patch(monkeypatch, dispute)

    initial = _snap(booking, dispute)
    _assert_blocked(initial)

    bad_payer = svc.carrier_respond(
        booking,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="partner",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert bad_payer.ok is False
    assert bad_payer.status_code == 400
    assert booking.amount == 40.0
    assert booking.billed_to_type == "clinic"

    responded = svc.carrier_respond(
        booking,
        stance="needs_correction",
        proposed_amount_ht=35,
        proposed_payer_type="patient",
        proposed_correction_note="mauvais payeur",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert responded.ok is True
    mid = _snap(booking, dispute)
    assert mid["status"] == "awaiting_correction"
    assert mid["stance"] == "needs_correction"
    assert mid["who_may_act"] == ("COMPANY",)
    assert mid["proposed_amount_ht"] == 35.0
    assert mid["proposed_payer_type"] == "patient"
    assert mid["amount_ht"] == 40.0
    assert mid["payer"] == "clinic"
    _assert_blocked(mid)

    evidence = svc.add_carrier_evidence(
        booking,
        kind="institution_written",
        note="mail clinique",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert evidence.ok is True
    _assert_blocked(_snap(booking, dispute))
    assert booking.amount == 40.0
    assert booking.billed_to_type == "clinic"

    submitted = svc.submit_dispute_for_validation(
        booking, actor_user_id=2, actor_role="COMPANY"
    )
    assert submitted.ok is True
    pending = _snap(booking, dispute)
    assert pending["status"] == "evidence_submitted"
    assert pending["who_may_act"] == ("institution", "admin")
    assert pending["amount_ht"] == 40.0
    assert pending["payer"] == "clinic"
    _assert_blocked(pending)

    carrier_close = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note=None,
        actor_user_id=9,
        actor_role="COMPANY",
    )
    assert carrier_close.ok is False
    assert carrier_close.status_code == 403
    assert booking.amount == 40.0
    assert booking.billed_to_type == "clinic"

    accepted = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note="correction ok",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    assert accepted.ok is True
    final = _snap(booking, dispute)
    assert final["status"] == "resolved_carrier"
    assert final["terminal"] is True
    assert final["who_may_act"] == ()
    assert booking.amount == 35.0
    assert booking.billed_to_type == "patient"
    assert final["clinic_line_in_invoice"] is False
    assert final["bucket"] == "patient_billable"


def test_g2_branch_c_amount_only_stays_clinic_until_third_party(monkeypatch):
    booking = _booking()
    dispute = _dispute()
    _patch(monkeypatch, dispute)

    responded = svc.carrier_respond(
        booking,
        stance="needs_correction",
        proposed_amount_ht=40,
        proposed_payer_type="clinic",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert responded.ok is True
    _assert_blocked(_snap(booking, dispute))
    assert booking.amount == 40.0

    svc.add_carrier_evidence(
        booking,
        kind="signed_transport_sheet",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    svc.submit_dispute_for_validation(booking, actor_user_id=2, actor_role="COMPANY")
    pending = _snap(booking, dispute)
    assert pending["who_may_act"] == ("institution", "admin")
    _assert_blocked(pending)

    accepted = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note="montant confirmé",
        actor_user_id=8,
        actor_role="admin",
    )
    assert accepted.ok is True
    final = _snap(booking, dispute)
    assert final["status"] == "resolved_carrier"
    assert final["clinic_line_in_invoice"] is True
    assert final["bucket"] == "clinic_billable"
    assert final["payer"] == "clinic"


def test_g2_evidence_requires_explicit_stance(monkeypatch):
    booking = _booking()
    dispute = _dispute()
    _patch(monkeypatch, dispute)
    result = svc.add_carrier_evidence(
        booking,
        kind="signed_transport_sheet",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert result.ok is False
    assert result.status_code == 409
    _assert_blocked(_snap(booking, dispute))


def test_g2_submit_without_intermediate_status_is_409(monkeypatch):
    booking = _booking()
    dispute = _dispute(
        status="disputed",
        carrier_stance="mission_done",
        evidence=[SimpleNamespace(source="uploaded", kind="signed_transport_sheet")],
    )
    _patch(monkeypatch, dispute)
    result = svc.submit_dispute_for_validation(
        booking, actor_user_id=2, actor_role="COMPANY"
    )
    assert result.ok is False
    assert result.status_code == 409
    _assert_blocked(_snap(booking, dispute))

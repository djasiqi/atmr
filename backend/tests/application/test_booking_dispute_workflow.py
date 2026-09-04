"""Workflow contestation — pas de DELETE, preuves, gel, décision tierce."""

from __future__ import annotations

from types import SimpleNamespace

from application.invoices.booking_dispute import freeze as freeze_mod
from application.invoices.booking_dispute import service as svc
from application.invoices.institution_invoice_reconciliation import (
    classify_booking_bucket,
)
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


def test_carrier_cannot_self_resolve():
    result = svc.decide_dispute(
        _booking(),
        decision="accept_carrier",
        note=None,
        actor_user_id=9,
        actor_role="COMPANY",
    )
    assert result.ok is False
    assert result.status_code == 403


def test_institution_right_marks_not_billable_without_delete(monkeypatch):
    booking = _booking()
    dispute = _dispute()
    monkeypatch.setattr(svc, "ensure_open_dispute", lambda *a, **k: dispute)
    monkeypatch.setattr(svc, "bump_edit_version", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_append_event", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_record_billing_event", lambda *_a, **_k: None)

    result = svc.confirm_institution_right(
        booking,
        exclusion_reason="created_by_error",
        note="saisie erronée",
        actor_user_id=2,
        actor_role="COMPANY",
    )
    assert result.ok is True
    assert dispute.status == "resolved_institution"
    assert booking.invoice_billing_status == "not_billable"
    assert not hasattr(booking, "deleted_at")
    assert classify_booking_bucket(booking)[0] == "other_excluded"


def test_submit_requires_uploaded_evidence(monkeypatch):
    booking = _booking()
    dispute = _dispute(
        carrier_stance="mission_done",
        evidence=[SimpleNamespace(source="system", kind="system_snapshot")],
    )
    monkeypatch.setattr(svc, "get_open_dispute", lambda *_a, **_k: dispute)
    result = svc.submit_dispute_for_validation(
        booking, actor_user_id=2, actor_role="COMPANY"
    )
    assert result.ok is False
    assert "justificatif" in (result.error or "").lower()


def test_accept_carrier_returns_billable(monkeypatch):
    booking = _booking()
    dispute = _dispute(
        status="evidence_submitted",
        carrier_stance="mission_done",
        evidence=[SimpleNamespace(source="uploaded", kind="signed_transport_sheet")],
    )
    monkeypatch.setattr(svc, "get_open_dispute", lambda *_a, **_k: dispute)
    monkeypatch.setattr(svc, "bump_edit_version", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_append_event", lambda *_a, **_k: None)
    monkeypatch.setattr(svc, "_record_billing_event", lambda *_a, **_k: None)

    result = svc.decide_dispute(
        booking,
        decision="accept_carrier",
        note="ok",
        actor_user_id=8,
        actor_role="institution_admin",
    )
    assert result.ok is True
    assert dispute.status == "resolved_carrier"
    assert booking.invoice_billing_status == "billable"
    assert (
        booking.institution_control_status == InstitutionBillingControlStatus.VALIDATED
    )


def test_financial_freeze_blocks_silent_change(monkeypatch):
    monkeypatch.setattr(
        freeze_mod, "get_open_dispute_for_booking", lambda *_a, **_k: object()
    )
    blocked, msg = freeze_mod.financial_change_blocked_by_dispute(_booking())
    assert blocked is True
    assert msg is not None
    assert "gelés" in msg


def test_disputed_stays_blocked_until_resolution():
    booking = _booking()
    bucket, reason = classify_booking_bucket(booking)
    assert bucket == "disputed_blocked"
    assert reason == "market_disputed"

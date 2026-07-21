"""Tests CancellationRespondPolicy V1 — OUTBOUND_ONLY + situations + montants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from application.institutions.cancellation_respond_policy import (
    BillingOutcome,
    CalculationCode,
    CancellationRespondError,
    CancellationRespondErrorCode,
    CancellationSituation,
    TripLeg,
    build_cancellation_respond_context,
    clear_cancellation_billing,
    compute_approach_fee,
    compute_full_cancellation_charge,
    get_cancellation_billing_booking,
    is_outbound_leg,
    parse_fee_amount,
    resolve_selected_fee,
    resolve_trip_leg,
    serialize_respond_ui,
)


def _booking(**kwargs):
    defaults = {
        "id": 1,
        "status": "ASSIGNED",
        "is_return": False,
        "parent_booking_id": None,
        "is_return_stop": False,
        "route_sequence_number": 1,
        "amount": Decimal("100.00"),
        "driver_id": 10,
        "company_id": 18,
        "executing_company_id": None,
        "scheduled_time": datetime.now(UTC) + timedelta(hours=48),
        "active_change_request_id": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _action(**kwargs):
    defaults = {
        "id": 99,
        "booking_id": 1,
        "action_type": "CANCELLATION",
        "action_scope": "BOOKING",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


SAMPLE_POLICY = {
    "enabled": True,
    "free_cancellation_threshold_hours": 24,
    "basis": "booking_amount",
    "apply_when_driver_assigned_only": False,
    "tiers": [
        {"id": "t24", "type": "time", "hours_before": 24, "percent": 30, "label": "< 24h"},
        {"id": "ten", "type": "status", "status": "EN_ROUTE", "percent": 70, "label": "EN_ROUTE"},
    ],
    "min_fee_chf": 0,
    "max_fee_chf": None,
}


def test_resolve_trip_leg_outbound_and_return():
    assert resolve_trip_leg(_booking()) == TripLeg.OUTBOUND
    assert is_outbound_leg(_booking()) is True
    assert resolve_trip_leg(_booking(is_return=True, parent_booking_id=1)) == TripLeg.RETURN
    assert is_outbound_leg(_booking(parent_booking_id=5)) is False


def test_get_cancellation_billing_booking_prefers_outbound():
    outbound = _booking(id=1, is_return=False)
    ret = _booking(id=2, is_return=True, parent_booking_id=1)
    assert get_cancellation_billing_booking([outbound, ret]) is outbound
    assert get_cancellation_billing_booking([ret]) is None


def test_compute_full_and_approach_refuse_return():
    ret = _booking(is_return=True, parent_booking_id=1)
    with pytest.raises(CancellationRespondError) as exc:
        compute_full_cancellation_charge(ret)
    assert exc.value.code == CancellationRespondErrorCode.BILLING_OUTCOME_NOT_ALLOWED

    with pytest.raises(CancellationRespondError):
        compute_approach_fee(ret, SAMPLE_POLICY)


def test_compute_full_outbound():
    q = compute_full_cancellation_charge(_booking(amount=Decimal("80.00")))
    assert q.amount == Decimal("80.00")
    assert q.calculation_code == CalculationCode.FULL_OUTBOUND_FARE


def test_clear_cancellation_billing():
    b = _booking(
        is_cancellation_billable=True,
        cancellation_fee_amount=Decimal("40.00"),
        cancellation_fee_percent=40,
        cancellation_fee_tier_id="t1",
    )
    clear_cancellation_billing(b)
    assert b.is_cancellation_billable is False
    assert b.cancellation_fee_amount == Decimal("0.00")
    assert b.cancellation_fee_percent is None
    assert b.cancellation_fee_tier_id is None


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_free_window_suggested_outcome(mock_policy, mock_affected):
    booking = _booking(scheduled_time=datetime.now(UTC) + timedelta(hours=48))
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")

    ctx = build_cancellation_respond_context(booking, action)
    assert ctx.situation == CancellationSituation.FREE_WINDOW
    assert ctx.primary_cta == "acknowledge_cancellation"
    assert ctx.suggested_outcome["calculation_code"] == CalculationCode.FREE_CANCELLATION_WINDOW
    assert ctx.suggested_outcome["amount"] == "0.00"
    ui = serialize_respond_ui(ctx)
    assert ui["billing_scope"] == "OUTBOUND_ONLY"


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_non_billable_return(mock_policy, mock_affected):
    ret = _booking(id=2, is_return=True, parent_booking_id=1, status="ASSIGNED")
    # aller déjà terminé → non cancelable ; seul retour cancelable
    outbound_done = _booking(
        id=1, status="COMPLETED", is_return=False, amount=Decimal("100")
    )
    outbound_done.is_cancellation_billable = True
    outbound_done.cancellation_fee_amount = Decimal("25.00")

    action = _action(booking_id=2)
    mock_affected.return_value = [outbound_done, ret]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")

    ctx = build_cancellation_respond_context(ret, action)
    assert ctx.situation == CancellationSituation.NON_BILLABLE_RETURN
    assert ctx.billing_eligible_booking_id is None
    assert ctx.billing_scope == "NONE"
    assert 1 in ctx.non_cancelable_booking_ids
    assert 2 in ctx.cancelable_booking_ids
    assert ctx.suggested_outcome["calculation_code"] == CalculationCode.NON_BILLABLE_RETURN


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_fee_window_outcomes(mock_policy, mock_affected):
    booking = _booking(
        scheduled_time=datetime.now(UTC) + timedelta(hours=6),
        amount=Decimal("100.00"),
    )
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")

    ctx = build_cancellation_respond_context(booking, action)
    assert ctx.situation == CancellationSituation.FEE_WINDOW
    codes = {o["code"] for o in ctx.allowed_outcomes}
    assert codes == {"ZERO", "POLICY_FEE", "CUSTOM"}
    assert ctx.primary_cta == "confirm_with_billing"


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_en_route_outcomes(mock_policy, mock_affected):
    booking = _booking(status="EN_ROUTE", amount=Decimal("100.00"))
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")

    ctx = build_cancellation_respond_context(booking, action)
    assert ctx.situation == CancellationSituation.EN_ROUTE
    codes = {o["code"] for o in ctx.allowed_outcomes}
    assert codes == {"ZERO", "APPROACH_FEE", "FULL_FARE", "CUSTOM"}


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_resolve_selected_fee_policy_fee_rejects_client_amount(mock_policy, mock_affected):
    booking = _booking(
        scheduled_time=datetime.now(UTC) + timedelta(hours=6),
        amount=Decimal("100.00"),
    )
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")
    ctx = build_cancellation_respond_context(booking, action)

    with pytest.raises(CancellationRespondError) as exc:
        resolve_selected_fee(
            ctx,
            billing_outcome="POLICY_FEE",
            fee_amount="50.00",
            body_provided=True,
        )
    assert exc.value.code == CancellationRespondErrorCode.FEE_AMOUNT_NOT_ALLOWED


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_resolve_selected_fee_custom_requires_comment_and_amount(
    mock_policy, mock_affected
):
    booking = _booking(
        scheduled_time=datetime.now(UTC) + timedelta(hours=6),
        amount=Decimal("100.00"),
    )
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")
    ctx = build_cancellation_respond_context(booking, action)

    with pytest.raises(CancellationRespondError) as exc:
        resolve_selected_fee(
            ctx,
            billing_outcome="CUSTOM",
            fee_amount=None,
            billing_comment="ok",
            body_provided=True,
        )
    assert exc.value.code == CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_REQUIRED

    with pytest.raises(CancellationRespondError) as exc2:
        resolve_selected_fee(
            ctx,
            billing_outcome="CUSTOM",
            fee_amount="50.00",
            billing_comment="",
            body_provided=True,
        )
    assert exc2.value.code == CancellationRespondErrorCode.BILLING_COMMENT_REQUIRED

    outcome, quote, comment = resolve_selected_fee(
        ctx,
        billing_outcome="CUSTOM",
        fee_amount="50.00",
        billing_comment="Chauffeur mobilisé",
        body_provided=True,
    )
    assert outcome == BillingOutcome.CUSTOM
    assert quote.amount == Decimal("50.00")
    assert comment == "Chauffeur mobilisé"


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_fee_window_requires_body(mock_policy, mock_affected):
    booking = _booking(scheduled_time=datetime.now(UTC) + timedelta(hours=6))
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")
    ctx = build_cancellation_respond_context(booking, action)

    with pytest.raises(CancellationRespondError) as exc:
        resolve_selected_fee(ctx, billing_outcome=None, body_provided=False)
    assert exc.value.code == CancellationRespondErrorCode.BILLING_OUTCOME_REQUIRED


@patch(
    "application.institutions.cancellation_respond_policy.resolve_affected_bookings"
)
@patch(
    "application.institutions.cancellation_respond_policy._load_cancellation_policy"
)
def test_free_window_implicit_zero(mock_policy, mock_affected):
    booking = _booking(scheduled_time=datetime.now(UTC) + timedelta(hours=48))
    action = _action()
    mock_affected.return_value = [booking]
    mock_policy.return_value = (SAMPLE_POLICY, "company-18-cancellation-v4")
    ctx = build_cancellation_respond_context(booking, action)
    outcome, quote, _ = resolve_selected_fee(
        ctx, billing_outcome=None, body_provided=False
    )
    assert outcome == BillingOutcome.ZERO
    assert quote.amount == Decimal("0.00")
    assert quote.calculation_code == CalculationCode.FREE_CANCELLATION_WINDOW


def test_parse_fee_amount_string_only():
    assert parse_fee_amount("50.00") == Decimal("50.00")
    with pytest.raises(CancellationRespondError) as exc:
        parse_fee_amount(50.0)
    assert exc.value.code == CancellationRespondErrorCode.CUSTOM_FEE_AMOUNT_INVALID


def test_effects_skip_non_cancelable():
    """Aller terminé intact + retour cancelable cleared."""
    from application.institutions.transport_action_workflow import (
        _apply_cancellation_effects,
    )
    from application.institutions.cancellation_respond_policy import FeeQuote

    outbound = MagicMock()
    outbound.id = 1
    outbound.status = "COMPLETED"
    outbound.is_cancellation_billable = True
    outbound.cancellation_fee_amount = Decimal("25.00")
    outbound.cancellation_fee_percent = 25
    outbound.cancellation_fee_tier_id = "hist"
    outbound.cancellation_reason_code = "CLIENT_REQUEST"
    outbound.cancellation_reason_text = "historique"
    outbound.cancellation_display_label = "hist"

    ret = MagicMock()
    ret.id = 2
    ret.status = "ASSIGNED"
    ret.is_return = True
    ret.parent_booking_id = 1

    action = MagicMock()
    action.id = 50
    action.reason = "Annulation retour"
    action.transport_request_id = None

    with patch(
        "application.institutions.cancellation_respond_policy.resolve_affected_bookings",
        return_value=[outbound, ret],
    ), patch(
        "application.institutions.transport_action_workflow._clear_driver_and_assignments"
    ):
        _apply_cancellation_effects(
            ret,
            action,
            reason="Annulation retour",
            cancelable_booking_ids=[2],
            billing_eligible_booking_id=None,
            fee_quote=FeeQuote(
                amount=Decimal("0.00"),
                calculation_code=CalculationCode.NON_BILLABLE_RETURN,
            ),
        )

    # aller historique inchangé
    assert outbound.status == "COMPLETED"
    assert outbound.cancellation_fee_amount == Decimal("25.00")
    assert outbound.is_cancellation_billable is True
    # retour annulé à zéro
    assert ret.status.value == "canceled" or str(ret.status).endswith("CANCELED") or ret.status == ret.status
    # MagicMock assignment: status set to BookingStatus.CANCELED enum
    from models.enums import BookingStatus

    assert ret.status == BookingStatus.CANCELED
    assert ret.is_cancellation_billable is False
    assert ret.cancellation_fee_amount == Decimal("0.00")

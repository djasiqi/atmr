"""Tests Phase A — statuts de livraison push canoniques."""

from __future__ import annotations

from services.notifications.push_delivery_status import (
    CONFIGURATION_ERROR,
    FAILED,
    INVALID_TOKEN,
    PROVIDER_ACCEPTED,
    PROVIDER_REJECTED,
    RECEIPT_ERROR,
    RECEIPT_NOT_APPLICABLE,
    RECEIPT_PENDING,
    RETRY_PENDING,
    apply_expo_receipt_to_classification,
    canonicalize_delivery_status,
    classify_push_result,
    ensure_deduplication_fields,
    sanitize_provider_text,
)


def test_canonicalize_aliases():
    assert canonicalize_delivery_status("sent") == PROVIDER_ACCEPTED
    assert canonicalize_delivery_status("rejected") == PROVIDER_REJECTED
    assert canonicalize_delivery_status("delivered") == "mobile_received"
    assert canonicalize_delivery_status("provider_accepted") == PROVIDER_ACCEPTED


def test_classify_fcm_ok():
    out = classify_push_result(
        {"ok": True, "message_id": "projects/x/messages/1"},
        provider="fcm",
    )
    assert out["delivery_status"] == PROVIDER_ACCEPTED
    assert out["provider_receipt_status"] == RECEIPT_NOT_APPLICABLE
    assert out["provider_message_id"] == "projects/x/messages/1"
    assert out["deactivate_token"] is False


def test_classify_expo_ok_receipt_pending():
    out = classify_push_result(
        {"ok": True, "provider_ticket_id": "ticket-abc"},
        provider="expo",
    )
    assert out["delivery_status"] == PROVIDER_ACCEPTED
    assert out["provider_receipt_status"] == RECEIPT_PENDING
    assert out["provider_ticket_id"] == "ticket-abc"


def test_classify_sender_mismatch_is_configuration_error():
    out = classify_push_result(
        {"ok": False, "error": "sender_id_mismatch", "configuration_error": True},
        provider="fcm",
    )
    assert out["delivery_status"] == CONFIGURATION_ERROR
    assert out["token_invalid"] is False
    assert out["deactivate_token"] is False


def test_classify_unregistered_invalid_token():
    out = classify_push_result(
        {"ok": False, "error": "token_unregistered", "token_invalid": True},
        provider="fcm",
    )
    assert out["delivery_status"] == INVALID_TOKEN
    assert out["deactivate_token"] is True


def test_classify_retry_exhausted_is_failed():
    out = classify_push_result(
        {"ok": False, "error": "retry_exhausted", "retry_exhausted": True},
        provider="expo",
    )
    assert out["delivery_status"] == FAILED
    assert out["failure_reason"] == "retry_exhausted"


def test_classify_network_retry_pending():
    out = classify_push_result(
        {"ok": False, "error": "Connection timeout", "retryable": True},
        provider="expo",
    )
    assert out["delivery_status"] == RETRY_PENDING


def test_expo_receipt_device_not_registered():
    base = classify_push_result(
        {"ok": True, "provider_ticket_id": "t1"},
        provider="expo",
    )
    updated = apply_expo_receipt_to_classification(
        base,
        receipt_status="error",
        receipt_error="DeviceNotRegistered",
    )
    assert updated["provider_receipt_status"] == RECEIPT_ERROR
    assert updated["delivery_status"] == INVALID_TOKEN
    assert updated["deactivate_token"] is True


def test_expo_receipt_invalid_credentials_config():
    base = classify_push_result(
        {"ok": True, "provider_ticket_id": "t1"},
        provider="expo",
    )
    updated = apply_expo_receipt_to_classification(
        base,
        receipt_status="error",
        receipt_error="InvalidCredentials",
    )
    assert updated["delivery_status"] == CONFIGURATION_ERROR
    assert updated["deactivate_token"] is False


def test_sanitize_redacts_expo_token():
    text = sanitize_provider_text("bad ExponentPushToken[xxxxxx] end")
    assert "ExponentPushToken" not in text
    assert "[REDACTED]" in text


def test_ensure_deduplication_fields():
    data = ensure_deduplication_fields(
        {"type": "booking_assigned", "booking_id": 42},
        driver_id=7,
    )
    assert "notification_id" in data
    assert "deduplication_key" in data
    assert data["dedupe_key"] == data["deduplication_key"]
    assert "booking_assigned:42:7:" in data["deduplication_key"]

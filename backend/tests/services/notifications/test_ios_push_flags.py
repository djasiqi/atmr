"""Tests flags iOS + fallback Expo (Phase B)."""

from __future__ import annotations

from services.notifications.ios_push_flags import (
    FAILURE_BEFORE_SEND,
    OUTCOME_UNKNOWN,
    allow_expo_fallback_for_fcm_issue,
    classify_fcm_issue_for_fallback,
    ios_disable_expo_on_fcm_upsert,
    ios_native_fcm_preferred,
)


def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("IOS_NATIVE_FCM_PREFERRED", raising=False)
    monkeypatch.delenv("IOS_DISABLE_EXPO_ON_FCM_UPSERT", raising=False)
    assert ios_native_fcm_preferred() is False
    assert ios_disable_expo_on_fcm_upsert() is False


def test_fallback_only_failure_before_send(monkeypatch):
    monkeypatch.setenv("IOS_EXPO_FALLBACK_ENABLED", "1")
    assert allow_expo_fallback_for_fcm_issue(FAILURE_BEFORE_SEND) is True
    assert allow_expo_fallback_for_fcm_issue(OUTCOME_UNKNOWN) is False
    assert allow_expo_fallback_for_fcm_issue("provider_accepted") is False


def test_classify_circuit_breaker_before_send():
    assert (
        classify_fcm_issue_for_fallback({"ok": False, "circuit_breaker_open": True})
        == FAILURE_BEFORE_SEND
    )


def test_classify_timeout_after_emit_unknown():
    assert (
        classify_fcm_issue_for_fallback(
            {"ok": False, "error": "timeout", "emitted": True}
        )
        == OUTCOME_UNKNOWN
    )

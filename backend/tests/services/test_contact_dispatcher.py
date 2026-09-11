from datetime import UTC, datetime

from services.contact.dispatcher import (
    build_contact_email_body,
    build_internal_subject,
    get_destination_email,
    get_sender_email,
    send_contact_notification,
)


def test_destination_and_sender_are_centralized(monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL_INSTITUTION", raising=False)
    monkeypatch.delenv("CONTACT_FROM_EMAIL_INSTITUTION", raising=False)
    monkeypatch.setenv("CONTACT_EMAIL_DEFAULT", "info@lirie.ch")
    monkeypatch.setenv("CONTACT_FROM_EMAIL", "noreply@lirie.ch")
    assert get_destination_email("institution") == "info@lirie.ch"
    assert get_sender_email("institution") == "noreply@lirie.ch"
    assert get_destination_email("transport") == "info@lirie.ch"
    assert get_sender_email("family") == "noreply@lirie.ch"


def test_per_category_env_is_ignored(monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL_INSTITUTION", "institution@lirie.ch")
    monkeypatch.setenv("CONTACT_FROM_EMAIL_INSTITUTION", "institution@lirie.ch")
    monkeypatch.setenv("CONTACT_EMAIL_DEFAULT", "info@lirie.ch")
    monkeypatch.setenv("CONTACT_FROM_EMAIL", "noreply@lirie.ch")
    assert get_destination_email("institution") == "info@lirie.ch"
    assert get_sender_email("institution") == "noreply@lirie.ch"


def test_internal_subject_includes_category_and_reference():
    subject = build_internal_subject("institution", "ct_XFWBW7A4DQAO")
    assert subject == (
        "[LIRIE Contact][Institution / Intégration] Nouvelle demande — ct_XFWBW7A4DQAO"
    )


def test_internal_body_contains_operational_fields():
    body = build_contact_email_body(
        {
            "trace_id": "ct_XFWBW7A4DQAO",
            "category": "institution",
            "name": "Drin JASIQI",
            "email": "jasiqi.drin@gmail.com",
            "phone": "+41790000000",
            "organization": "Clinique Test",
            "message": "Nous souhaitons intégrer LIRIE.",
            "created_at": datetime(2026, 9, 8, 11, 20, tzinfo=UTC),
        }
    )
    assert "Référence : ct_XFWBW7A4DQAO" in body
    assert "Type : Institution / Intégration" in body
    assert "Nom : Drin JASIQI" in body
    assert "Institution : Clinique Test" in body
    assert "Date : 08.09.2026 13:20" in body


def test_send_contact_notification_uses_noreply_info_and_reply_to(monkeypatch):
    calls = []

    def _fake_send(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return {"ok": True, "provider": "brevo"}

    monkeypatch.setattr(
        "services.contact.dispatcher.send_email_notification", _fake_send
    )
    monkeypatch.setenv("CONTACT_EMAIL_DEFAULT", "info@lirie.ch")
    monkeypatch.setenv("CONTACT_FROM_EMAIL", "noreply@lirie.ch")

    result = send_contact_notification(
        {
            "category": "institution",
            "name": "Drin JASIQI",
            "email": "jasiqi.drin@gmail.com",
            "trace_id": "ct_XFWBW7A4DQAO",
            "message": "Bonjour",
        }
    )

    assert result["internal_ok"] is True
    assert result["auto_reply_ok"] is True
    assert result["destination"] == "info@lirie.ch"
    assert result["from_email"] == "noreply@lirie.ch"
    assert result["reply_to"] == "jasiqi.drin@gmail.com"
    assert len(calls) == 2

    confirmation = calls[0]
    internal = calls[1]
    assert confirmation["args"][0] == "jasiqi.drin@gmail.com"
    assert "Confirmation de réception" in confirmation["args"][1]
    assert "Institution / Intégration" in confirmation["args"][1]
    assert "Nous avons bien reçu" in confirmation["args"][2]
    assert "Notre équipe" in confirmation["args"][2]
    assert "Référence :" in confirmation["args"][2]
    assert confirmation["kwargs"]["from_email"] == "noreply@lirie.ch"

    assert internal["args"][0] == "info@lirie.ch"
    assert internal["kwargs"]["from_email"] == "noreply@lirie.ch"
    assert internal["kwargs"]["reply_to"] == "jasiqi.drin@gmail.com"


def test_confirmation_is_sent_even_if_internal_fails(monkeypatch):
    calls = []

    def _fake_send(email, subject, body, notification_type="unknown", **kwargs):
        calls.append(notification_type)
        if notification_type == "contact_request":
            return {"ok": False, "error": "hard_bounce"}
        return {"ok": True, "provider": "brevo"}

    monkeypatch.setattr(
        "services.contact.dispatcher.send_email_notification", _fake_send
    )
    result = send_contact_notification(
        {
            "category": "institution",
            "email": "jasiqi.drin@gmail.com",
            "trace_id": "ct_FAILTEST",
            "message": "Bonjour",
        }
    )
    assert result["auto_reply_ok"] is True
    assert result["internal_ok"] is False
    assert result["ok"] is False
    assert calls == ["contact_autoreply", "contact_request"]

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from services.contact.retry import (
    MAX_NOTIFICATION_RETRIES,
    retry_failed_internal_notifications,
    retry_internal_notification,
)
from services.contact.status import apply_contact_notification_result


def _row(**overrides):
    values = {
        "id": 1,
        "status": "new",
        "email_delivery_status": "failed",
        "autoreply_delivery_status": "sent",
        "notification_retry_count": 0,
        "notification_last_error": "hard_bounce",
        "notification_last_attempt_at": datetime.now(UTC) - timedelta(minutes=10),
        "assigned_channel": "institution@lirie.ch",
        "category": "institution",
        "name": "Drin",
        "email": "drin@example.com",
        "phone": None,
        "organization": "EMS",
        "message": "Bonjour",
        "priority": "medium",
        "payload_json": {},
        "trace_id": "ct_RETRY1",
        "user_id": None,
        "user_role": None,
        "created_at": datetime.now(UTC),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_apply_result_keeps_request_on_internal_failure():
    row = _row(email_delivery_status="sending", status="new")
    apply_contact_notification_result(
        row,
        {
            "internal_ok": False,
            "internal_error": "smtp rejected",
            "auto_reply_ok": True,
            "destination": "info@lirie.ch",
        },
    )
    assert row.email_delivery_status == "failed"
    assert row.status == "new"
    assert row.autoreply_delivery_status == "sent"
    assert row.assigned_channel == "info@lirie.ch"
    assert row.notification_last_error == "smtp rejected"


def test_retry_internal_notification_recovers(monkeypatch):
    row = _row()
    monkeypatch.setattr(
        "services.contact.retry.send_internal_contact_notification",
        lambda _payload: {
            "ok": True,
            "destination": "info@lirie.ch",
            "from_email": "noreply@lirie.ch",
        },
    )
    result = retry_internal_notification(row)
    assert result["ok"] is True
    assert row.email_delivery_status == "sent"
    assert row.status == "triaged"
    assert row.notification_retry_count == 0


def test_retry_skips_exhausted_attempts():
    row = _row(notification_retry_count=MAX_NOTIFICATION_RETRIES)
    result = retry_internal_notification(row, ignore_cooldown=True)
    assert result["skipped"] is True
    assert row.email_delivery_status == "failed"


def test_retry_batch_commits(monkeypatch):
    rows = [_row(), _row(id=2, trace_id="ct_RETRY2")]
    store = {"committed": False}

    class _Query:
        def filter(self, *args, **kwargs):
            return self

        def order_by(self, *args, **kwargs):
            return self

        def limit(self, *_args):
            return self

        def all(self):
            return rows

    class _Column:  # noqa: PLW1641
        def __eq__(self, _other):
            return self

        def __ne__(self, _other):
            return self

        def __lt__(self, _other):
            return self

        def asc(self):
            return self

    class _ContactRequest:
        query = _Query()
        email_delivery_status = _Column()
        status = _Column()
        notification_retry_count = _Column()
        created_at = _Column()

    monkeypatch.setattr("services.contact.retry.ContactRequest", _ContactRequest)
    monkeypatch.setattr(
        "services.contact.retry.send_internal_contact_notification",
        lambda payload: {
            "ok": payload["trace_id"] == "ct_RETRY1",
            "error": None if payload["trace_id"] == "ct_RETRY1" else "still_failing",
            "destination": "info@lirie.ch",
        },
    )
    monkeypatch.setattr(
        "services.contact.retry.db.session.commit",
        lambda: store.update(committed=True),
    )
    summary = retry_failed_internal_notifications()
    assert summary["attempted"] == 2
    assert summary["recovered"] == 1
    assert store["committed"] is True

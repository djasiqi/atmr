"""Pied de page et dates des PDF de rappel."""

from __future__ import annotations

from datetime import UTC, datetime

from services.documents.pdf import _get_reminder_footer_message


def test_reminder_footer_level_2_includes_days_and_deadline():
    due = datetime(2026, 6, 27, tzinfo=UTC)
    message = _get_reminder_footer_message(2, payment_days=10, due_date=due)

    assert "sous 10 jours" in message
    assert "27.06.2026" in message
    assert "meilleurs délais" not in message

# tests/services/notifications/test_institution_new_request_push_gate2.py
"""GATE 2 — push new_request uniquement si inbox persistée (pas de doublon FCM)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tasks import request_offer_tasks


def test_notify_company_new_offer_skips_push_when_inbox_deduped():
    transport_request = SimpleNamespace(
        id=456,
        public_id="pub-456",
        institution=SimpleNamespace(name="Clinique"),
        patient=SimpleNamespace(first_name="Jean", last_name="Dupont"),
        is_round_trip=False,
    )

    with (
        patch(
            "services.events.institution_events.persist_company_notification",
            return_value=None,
        ) as mock_persist,
        patch(
            "services.notifications.institution_new_request_push.enqueue_institution_new_request_company_push"
        ) as mock_enqueue,
        patch("models.Company"),
        patch(
            "services.demo.soft_delete_guard.institution_is_demo",
            return_value=False,
        ),
        patch(
            "services.institutions.mission_schedule.get_effective_dispatch_time",
            return_value=SimpleNamespace(strftime=lambda _fmt: "14:00"),
        ),
        patch(
            "services.institutions.mission_schedule.get_mission_date",
            return_value=None,
        ),
    ):
        request_offer_tasks._notify_company_new_offer(
            transport_request,
            company_id=1,
            offer_id=123,
        )

    mock_persist.assert_called_once()
    mock_enqueue.assert_not_called()

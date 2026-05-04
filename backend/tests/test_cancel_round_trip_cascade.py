# ruff: noqa: I001
"""Tests: R5 — annulation cascade aller → retour.

Vérifie que l'annulation de l'aller d'un A/R annule automatiquement le retour
(non facturable) via le handler centralisé, avec guard anti-boucle.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class _FakeBooking:
    id: int
    company_id: int = 5
    driver_id: int | None = None
    status: str = "PENDING"
    is_round_trip: bool = False
    is_return: bool = False
    parent_booking_id: int | None = None
    return_trip: _FakeBooking | None = None
    cancelled_at: datetime | None = None
    cancelled_by_role: str | None = None
    cancellation_reason_code: str | None = None
    cancellation_reason_text: str | None = None
    is_cancellation_billable: bool | None = None
    cancellation_display_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "status": self.status}


def _build_round_trip_pair(
    outbound_status: str = "CANCELED",
    return_status: str = "PENDING",
) -> tuple[_FakeBooking, _FakeBooking]:
    ret = _FakeBooking(
        id=2,
        status=return_status,
        is_return=True,
        parent_booking_id=1,
    )
    outbound = _FakeBooking(
        id=1,
        status=outbound_status,
        is_round_trip=True,
        is_return=False,
        return_trip=ret,
    )
    return outbound, ret


def _make_cancel_fields() -> dict[str, Any]:
    return {
        "cancelled_at": datetime.now(UTC),
        "cancelled_by_role": "system",
        "cancellation_reason_code": "OUTBOUND_CANCELLED",
        "cancellation_reason_text": "Retour annulé automatiquement (aller annulé)",
        "is_cancellation_billable": False,
        "cancellation_display_label": "Retour annulé (aller annulé)",
    }


class _FakeBookingStatus:
    CANCELED = "CANCELED"
    EN_ROUTE = "EN_ROUTE"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    RETURN_COMPLETED = "RETURN_COMPLETED"


def _run_handler(event: dict[str, Any], booking: _FakeBooking | None) -> list[Any]:
    """Run handle_booking_cancelled with full patching. Returns published events."""
    published: list[Any] = []

    mock_db = MagicMock()
    mock_db.session.get.return_value = booking
    mock_db.session.rollback = MagicMock()
    mock_db.session.flush = MagicMock()

    with (
        patch("ext.db", mock_db),
        patch("models.Booking", MagicMock()),
        patch("models.enums.BookingStatus", _FakeBookingStatus),
        patch(
            "application.bookings.cancellation_rules.compute_cancellation_fields",
            return_value=_make_cancel_fields(),
        ),
        patch("application.bookings.cancellation_rules.log_cancellation_persisted"),
        patch(
            "shared.events.event_bus.publish_event",
            side_effect=lambda e: published.append(e),
        ),
        patch("services.notifications.core.notify_booking_cancelled"),
        patch(
            "services.notifications.notification_targets.resolve_booking_notification_context",
            return_value=None,
        ),
        patch(
            "services.notifications.notification_targets.compute_all_notification_targets",
        ),
    ):
        # Force-reload to pick up patches
        import importlib
        import services.events.handlers.booking_handlers as mod

        importlib.reload(mod)
        mod.handle_booking_cancelled(event)

    return published


class TestCancelOutboundCascadesReturn:
    """Annuler l'aller d'un A/R annule automatiquement le retour."""

    def test_cancel_outbound_cancels_return(self):
        outbound, ret = _build_round_trip_pair()
        published = _run_handler(
            {"booking_id": 1, "actor_role": "company", "actor_id": 5},
            outbound,
        )

        assert ret.status == "CANCELED"
        assert ret.is_cancellation_billable is False
        assert ret.cancellation_reason_code == "OUTBOUND_CANCELLED"
        assert len(published) == 1
        evt = published[0]
        assert evt.booking_id == 2
        assert evt.cancel_source == "cascade_from_outbound"

    def test_cancel_outbound_marks_return_not_billable(self):
        outbound, ret = _build_round_trip_pair()
        _run_handler(
            {"booking_id": 1, "actor_role": "company"},
            outbound,
        )

        assert ret.is_cancellation_billable is False

    def test_cascade_does_not_loop(self):
        """Event avec cancel_source=cascade_from_outbound ne re-cascade pas."""
        ret = _FakeBooking(id=2, status="CANCELED", is_return=True, parent_booking_id=1)
        published = _run_handler(
            {
                "booking_id": 2,
                "actor_role": "system",
                "cancel_source": "cascade_from_outbound",
            },
            ret,
        )

        assert len(published) == 0

    def test_cancel_return_does_not_cancel_outbound(self):
        """Annuler le retour seul ne touche pas l'aller."""
        outbound = _FakeBooking(id=1, status="COMPLETED", is_round_trip=True)
        ret = _FakeBooking(
            id=2,
            status="CANCELED",
            is_return=True,
            is_round_trip=False,
            parent_booking_id=1,
        )

        published = _run_handler(
            {"booking_id": 2, "actor_role": "company"},
            ret,
        )

        assert outbound.status == "COMPLETED"
        assert len(published) == 0

    def test_cancel_outbound_when_return_already_cancelled(self):
        """Si le retour est déjà annulé, pas de re-traitement."""
        outbound, ret = _build_round_trip_pair(return_status="CANCELED")
        published = _run_handler(
            {"booking_id": 1, "actor_role": "company"},
            outbound,
        )

        assert len(published) == 0
        assert ret.status == "CANCELED"

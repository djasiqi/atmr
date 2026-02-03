"""Tests pour DeleteOrCancelCompanyReservationUseCase (annulation standardisée)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from application.companies.reservations.delete_or_cancel_reservation import (
    DeleteOrCancelCompanyReservationUseCase,
)


class FakeBooking:
    """Booking mock avec champs d'annulation."""

    def __init__(
        self,
        *,
        id: int = 1,
        status: str = "ASSIGNED",
        driver_id: int | None = 1,
        scheduled_time: datetime | None = None,
    ):
        self.id = id
        self.status = status
        self.driver_id = driver_id
        self.scheduled_time = scheduled_time or (datetime.now(UTC) + timedelta(hours=2))
        self.cancelled_at = None
        self.cancelled_by_role = None
        self.cancellation_reason_code = None
        self.cancellation_reason_text = None
        self.is_cancellation_billable = None
        self.cancellation_display_label = None


def test_cancel_assigned_persists_cancellation_fields_without_reason():
    """Sans body → reason_code=None → Annulation (historique), non facturé."""
    booking = FakeBooking(status="ASSIGNED", driver_id=1)
    uc = DeleteOrCancelCompanyReservationUseCase()
    result = uc.execute(
        booking,
        now_utc=datetime.now(UTC),
        hours_offset=-24.0,
        reason_code=None,
        reason_text=None,
    )
    assert result.ok is True
    assert result.action == "cancel"
    assert str(booking.status).upper() == "CANCELED"
    assert booking.driver_id is None
    assert booking.cancelled_at is not None
    assert booking.cancelled_by_role == "company"
    assert booking.cancellation_reason_code == "OTHER"
    assert booking.is_cancellation_billable is False
    assert booking.cancellation_display_label == "Annulation (historique)"


def test_cancel_assigned_persists_cancellation_fields_with_reason():
    """Avec reason_code=NO_SHOW → facturé, libellé correct."""
    booking = FakeBooking(status="ASSIGNED", driver_id=1)
    uc = DeleteOrCancelCompanyReservationUseCase()
    result = uc.execute(
        booking,
        now_utc=datetime.now(UTC),
        hours_offset=-24.0,
        reason_code="NO_SHOW",
        reason_text=None,
    )
    assert result.ok is True
    assert result.action == "cancel"
    assert str(booking.status).upper() == "CANCELED"
    assert booking.cancelled_by_role == "company"
    assert booking.cancellation_reason_code == "NO_SHOW"
    assert booking.is_cancellation_billable is True
    assert booking.cancellation_display_label == "Client ne s'est pas présenté"
    assert result.is_cancellation_billable is True
    assert result.cancellation_display_label == "Client ne s'est pas présenté"


def test_cancel_assigned_with_company_issue_non_billable():
    """COMPANY_ISSUE → non facturé."""
    booking = FakeBooking(status="ASSIGNED", driver_id=1)
    uc = DeleteOrCancelCompanyReservationUseCase()
    result = uc.execute(
        booking,
        reason_code="COMPANY_ISSUE",
        reason_text=None,
    )
    assert result.ok is True
    assert result.action == "cancel"
    assert booking.is_cancellation_billable is False
    assert booking.cancellation_display_label == "Problème entreprise"


def test_delete_pending_does_not_set_cancellation_fields():
    """PENDING → delete, pas de champs annulation."""
    booking = FakeBooking(status="PENDING", driver_id=None)
    uc = DeleteOrCancelCompanyReservationUseCase()
    result = uc.execute(booking, reason_code=None, reason_text=None)
    assert result.ok is True
    assert result.action == "delete"
    assert booking.cancelled_at is None
    assert booking.cancellation_reason_code is None

# ruff: noqa: I001
"""Tests Phase 1 — migration sentinelle 00:00 (null + time_confirmed explicite)."""

from __future__ import annotations

from datetime import datetime

import pytest

from models import Booking, BookingStatus
from repositories.booking_repository import BookingRepository
from shared.time_utils import day_local_bounds


class TestFindForDispatchExcludesUnconfirmedNull:
    """Retours/legs sans heure confirmée exclus du dispatch auto."""

    def test_find_for_dispatch_excludes_null_unconfirmed(self, db, sample_company):
        company = sample_company
        repo = BookingRepository()
        day = "2026-06-11"
        day_start, _ = day_local_bounds(day)

        outbound = Booking(
            company_id=company.id,
            customer_name="Patient Test",
            pickup_location="A",
            dropoff_location="B",
            scheduled_time=day_start.replace(hour=8),
            time_confirmed=True,
            status=BookingStatus.ACCEPTED.value,
            amount=50.0,
            is_return=False,
        )
        db.session.add(outbound)
        db.session.flush()

        pending_return = Booking(
            company_id=company.id,
            customer_name="Patient Test",
            pickup_location="B",
            dropoff_location="A",
            scheduled_time=None,
            time_confirmed=False,
            status=BookingStatus.ACCEPTED.value,
            amount=50.0,
            is_return=True,
            parent_booking_id=outbound.id,
        )
        db.session.add(pending_return)
        db.session.commit()

        dispatch_ids = {
            b.id for b in repo.find_for_dispatch(company.id, horizon_minutes=24 * 60)
        }
        assert outbound.id in dispatch_ids
        assert pending_return.id not in dispatch_ids


class TestDashboardDayFilterReturnDateOnly:
    """Visibilité dashboard : retour sans heure lié à l'aller du jour."""

    def test_return_visible_on_outbound_day_not_next_day(self, db, sample_company):
        from routes.companies import _reservations_base_query_for_company_day

        company = sample_company
        day_outbound = "2026-06-11"
        day_next = "2026-06-12"
        start_out, _ = day_local_bounds(day_outbound)

        outbound = Booking(
            company_id=company.id,
            customer_name="Patient Test",
            pickup_location="A",
            dropoff_location="B",
            scheduled_time=start_out.replace(hour=8),
            time_confirmed=True,
            status=BookingStatus.ACCEPTED.value,
            amount=50.0,
            is_return=False,
        )
        db.session.add(outbound)
        db.session.flush()

        pending_return = Booking(
            company_id=company.id,
            customer_name="Patient Test",
            pickup_location="B",
            dropoff_location="A",
            scheduled_time=None,
            time_confirmed=False,
            status=BookingStatus.ACCEPTED.value,
            amount=50.0,
            is_return=True,
            parent_booking_id=outbound.id,
        )
        db.session.add(pending_return)
        db.session.commit()

        ids_outbound_day = {
            b.id for b in _reservations_base_query_for_company_day(company.id, day_outbound).all()
        }
        ids_next_day = {
            b.id for b in _reservations_base_query_for_company_day(company.id, day_next).all()
        }

        assert outbound.id in ids_outbound_day
        assert pending_return.id in ids_outbound_day
        assert pending_return.id not in ids_next_day


class TestScheduleReservationExplicitTimeConfirmed:
    """Planification : time_confirmed explicite (heure ajoutée après création)."""

    def test_schedule_sets_explicit_time_confirmed(self):
        from application.companies.reservations.schedule_reservation import (
            ScheduleCompanyReservationUseCase,
        )

        booking = Booking(
            customer_name="Test",
            pickup_location="A",
            dropoff_location="B",
            scheduled_time=None,
            time_confirmed=False,
            status=BookingStatus.ACCEPTED.value,
            amount=50.0,
            is_return=True,
        )

        uc = ScheduleCompanyReservationUseCase()
        result = uc.execute(
            booking,
            scheduled_time_iso="2026-06-11T14:30:00",
            time_confirmed=True,
        )
        assert result.ok is True
        assert booking.scheduled_time is not None
        assert booking.scheduled_time.hour == 14
        assert booking.time_confirmed is True

    def test_schedule_syncs_transport_request_departure(self, monkeypatch):
        from application.companies.reservations.schedule_reservation import (
            ScheduleCompanyReservationUseCase,
        )

        synced: list[int] = []

        def _fake_sync(booking):
            synced.append(getattr(booking, "id", 0))
            return True

        monkeypatch.setattr(
            "services.institutions.mission_schedule.sync_request_departure_for_booking",
            _fake_sync,
        )

        booking = Booking(
            id=35204,
            customer_name="Test",
            pickup_location="A",
            dropoff_location="B",
            scheduled_time=None,
            time_confirmed=False,
            status=BookingStatus.ACCEPTED.value,
            amount=50.0,
        )
        uc = ScheduleCompanyReservationUseCase()
        result = uc.execute(
            booking,
            scheduled_time_iso="2026-06-17T10:00:00",
            time_confirmed=True,
        )
        assert result.ok is True
        assert synced == [35204]

"""Tests d'intégration : agrégation mensuelle des bookings et endpoint admin/stats."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from dateutil.relativedelta import relativedelta

from models import Booking
from models.enums import BookingStatus
from repositories.booking_repository import BookingRepository


def _make_booking(
    db,
    *,
    company_id: int,
    client_id: int,
    user_id: int,
    created_at: datetime,
) -> Booking:
    """Booking minimal avec created_at contrôlé (tests bornes / agrégation)."""
    b = Booking()
    b.user_id = user_id
    b.company_id = company_id
    b.client_id = client_id
    b.customer_name = "Stats Test"
    b.pickup_location = "Rue A, 1000 Lausanne"
    b.dropoff_location = "Rue B, 1000 Lausanne"
    b.scheduled_time = created_at + timedelta(days=1)
    b.status = BookingStatus.PENDING
    b.amount = Decimal("10.00")
    b.vat_rate = Decimal("7.70")
    b.created_at = created_at
    db.session.add(b)
    db.session.flush()
    return b


@pytest.mark.integration
class TestGetMonthlyBookingCounts:
    """Verrouille get_monthly_booking_counts (PostgreSQL + date_trunc)."""

    def test_aggregates_two_months_and_keys_yyyy_mm(
        self,
        db,
        test_company,
        test_client,
        requires_postgresql,
    ):
        if not test_company or not test_client:
            pytest.skip("fixtures requis")
        uid = test_client.user_id
        assert uid is not None
        jan = datetime(2025, 1, 15, 12, 30, 0, tzinfo=UTC)
        feb = datetime(2025, 2, 10, 8, 0, 0, tzinfo=UTC)
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=jan,
        )
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=jan,
        )
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=feb,
        )

        repo = BookingRepository()
        start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        end = datetime(2025, 2, 28, 23, 59, 59, tzinfo=UTC)
        counts = repo.get_monthly_booking_counts(start, end)

        assert counts.get("2025-01") == 2
        assert counts.get("2025-02") == 1
        assert "2025-03" not in counts

    def test_same_day_multiple_bookings(
        self,
        db,
        test_company,
        test_client,
        requires_postgresql,
    ):
        if not test_company or not test_client:
            pytest.skip("fixtures requis")
        uid = test_client.user_id
        day = datetime(2025, 6, 3, 14, 0, 0, tzinfo=UTC)
        for _ in range(3):
            _make_booking(
                db,
                company_id=test_company.id,
                client_id=test_client.id,
                user_id=uid,
                created_at=day,
            )
        repo = BookingRepository()
        counts = repo.get_monthly_booking_counts(
            datetime(2025, 6, 1, tzinfo=UTC),
            datetime(2025, 6, 30, 23, 59, 59, tzinfo=UTC),
        )
        assert counts.get("2025-06") == 3

    def test_boundaries_inclusive_and_exclusion_outside_window(
        self,
        db,
        test_company,
        test_client,
        requires_postgresql,
    ):
        if not test_company or not test_client:
            pytest.skip("fixtures requis")
        uid = test_client.user_id
        # Inclus : premier instant du mois fenêtre
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC),
        )
        # Exclu : avant window_start
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=datetime(2025, 3, 31, 23, 59, 59, tzinfo=UTC),
        )
        # Inclus : borne haute end (<= end)
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=datetime(2025, 4, 30, 23, 59, 59, tzinfo=UTC),
        )
        repo = BookingRepository()
        counts = repo.get_monthly_booking_counts(
            datetime(2025, 4, 1, 0, 0, 0, tzinfo=UTC),
            datetime(2025, 4, 30, 23, 59, 59, tzinfo=UTC),
        )
        assert counts.get("2025-04") == 2


@pytest.mark.integration
class TestAdminStatsEndpointBookingTrends:
    """Forme et ordre des bookingTrends sur GET /api/v1/admin/stats."""

    def test_booking_trends_twelve_months_shape(
        self,
        client,
        admin_headers,
        requires_postgresql,
    ):
        response = client.get(
            "/api/v1/admin/stats",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        if response.status_code == 403:
            pytest.skip("IP whitelist ou JWT : ajuster environnement de test")
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json()
        assert data is not None
        trends = data.get("bookingTrends")
        assert isinstance(trends, list)
        assert len(trends) == 12
        now = datetime.now(UTC)
        current_month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        expected_months = [
            (current_month_start - relativedelta(months=i)).strftime("%Y-%m")
            for i in range(11, -1, -1)
        ]
        for i, item in enumerate(trends):
            assert set(item.keys()) >= {"month", "bookings"}
            assert item["month"] == expected_months[i]
            assert isinstance(item["bookings"], int)
            assert item["bookings"] >= 0

    def test_booking_trends_reflects_seeded_bookings(
        self,
        client,
        admin_headers,
        db,
        test_company,
        test_client,
        requires_postgresql,
    ):
        """Un booking créé ce mois-ci augmente le compteur du mois courant."""
        if not test_company or not test_client:
            pytest.skip("fixtures requis")
        uid = test_client.user_id
        assert uid is not None
        now = datetime.now(UTC)
        _make_booking(
            db,
            company_id=test_company.id,
            client_id=test_client.id,
            user_id=uid,
            created_at=now - timedelta(minutes=5),
        )
        db.session.flush()

        response = client.get(
            "/api/v1/admin/stats",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        if response.status_code == 403:
            pytest.skip("IP whitelist ou JWT : ajuster environnement de test")
        assert response.status_code == 200
        data = response.get_json()
        trends = data["bookingTrends"]
        current_month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        current_key = current_month_start.strftime("%Y-%m")
        current_entry = next(t for t in trends if t["month"] == current_key)
        assert current_entry["bookings"] >= 1

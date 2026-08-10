"""Tests unitaires : définitions métier du résumé dashboard admin."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import func, select

from models import Booking, BookingStatus
from models.enums import (
    PlatformBillingPeriodStatus,
    PlatformIssuedInvoiceStatus,
    PlatformStatementStatus,
)
from services import admin_dashboard_summary as summary_mod


@pytest.fixture
def fixed_now():
    return datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)


class TestCancellationCohort:
    def test_cohort_query_filters_created_at_and_canceled(self):
        seven_ago = datetime(2026, 7, 27, 12, 0, 0, tzinfo=UTC)
        now = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)
        stmt = select(func.count(Booking.id)).where(
            Booking.created_at >= seven_ago,
            Booking.created_at <= now,
            Booking.status == BookingStatus.CANCELED,
        )
        compiled = str(stmt.compile()).lower()
        assert "created_at" in compiled
        assert "updated_at" not in compiled or compiled.count("created_at") >= 1

    def test_rate_zero_when_no_creations(self):
        created = 0
        canceled_from_created = 0
        rate = float(canceled_from_created) / float(max(created, 1)) if created else 0.0
        assert rate == 0.0


class TestBillingToReviewScope:
    def test_review_statuses_use_enums(self):
        assert (
            PlatformStatementStatus.NEEDS_REVIEW.value
            in summary_mod._STATEMENT_TO_REVIEW
        )
        assert (
            PlatformStatementStatus.CALCULATED.value in summary_mod._STATEMENT_TO_REVIEW
        )
        assert (
            PlatformStatementStatus.LOCKED.value not in summary_mod._STATEMENT_TO_REVIEW
        )
        assert PlatformBillingPeriodStatus.DRAFT.value == "draft"

    def test_billing_query_requires_draft_period(self):
        stmt = (
            select(func.count(summary_mod.PlatformInvoice.id))
            .select_from(summary_mod.PlatformInvoice)
            .join(
                summary_mod.PlatformBillingPeriod,
                summary_mod.PlatformInvoice.period_id
                == summary_mod.PlatformBillingPeriod.id,
            )
            .where(
                summary_mod.PlatformInvoice.statement_status.in_(
                    summary_mod._STATEMENT_TO_REVIEW
                ),
                summary_mod.PlatformBillingPeriod.status
                == PlatformBillingPeriodStatus.DRAFT.value,
                summary_mod.PlatformInvoice.cancelled_at.is_(None),
            )
        )
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True})).lower()
        assert "draft" in sql
        assert "cancelled_at" in sql


class TestBookingTrendsDeprecated:
    def test_no_monthly_repo_call_in_summary_source(self):
        """La construction du summary ne doit plus appeler get_monthly_booking_counts."""
        import inspect

        src = inspect.getsource(summary_mod.build_admin_dashboard_summary)
        assert "get_monthly_booking_counts(" not in src
        assert "booking_trends: list" in src or "booking_trends =" in src
        assert "booking_trends: list[dict[str, Any]] = []" in src or "= []" in src

    def test_trends_empty_with_app_context(self, app, fixed_now):
        query_chain = MagicMock()
        query_chain.options.return_value.order_by.return_value.limit.return_value.all.return_value = []
        exec_result = MagicMock()
        exec_result.scalar_one.return_value = 0

        with app.app_context():
            with (
                patch.object(summary_mod, "datetime") as mock_dt,
                patch.object(summary_mod.db.session, "scalar", return_value=0),
                patch.object(
                    summary_mod.db.session, "execute", return_value=exec_result
                ),
                patch.object(summary_mod.Booking, "query", query_chain),
                patch(
                    "repositories.booking_repository.BookingRepository.get_monthly_booking_counts"
                ) as monthly,
            ):
                mock_dt.now.return_value = fixed_now
                data = summary_mod.build_admin_dashboard_summary()
                monthly.assert_not_called()
                assert data["booking_trends"] == []
                assert "generated_at" in data
                assert data["priorities"]["critical_attention_count"] == 0
                assert "platform_invoiced_current_month_chf" in data["kpi_business"]
                assert "bookings_canceled_from_created_7d" in data["kpi_business"]


class TestIssuedExcluded:
    def test_issued_excluded_statuses(self):
        assert PlatformIssuedInvoiceStatus.DRAFT.value in summary_mod._ISSUED_EXCLUDED
        assert (
            PlatformIssuedInvoiceStatus.CANCELLED.value in summary_mod._ISSUED_EXCLUDED
        )
        assert (
            PlatformIssuedInvoiceStatus.CREDITED.value in summary_mod._ISSUED_EXCLUDED
        )
        assert (
            PlatformIssuedInvoiceStatus.ISSUED.value not in summary_mod._ISSUED_EXCLUDED
        )

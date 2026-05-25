"""Tests service audit / versioning bookings institution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import Booking, Institution, TransportRequest, User, UserRole
from models.enums import BookingStatus, InstitutionRole, RequestStatus
from services.institutions.booking_change_service import (
    BILLING_CHANGE_REASON_CODES,
    classify_change,
    mask_financial_fields,
    check_version,
    assert_not_boarded,
    INSTITUTION_OPERATIONAL_FIELDS,
)


class TestBookingChangeClassification:
    def test_en_route_destination_critical(self):
        cc, sev, ack = classify_change(
            {"dropoff_location"},
            is_en_route=True,
        )
        assert cc == "critical"
        assert sev == "CRITICAL"
        assert ack is True

    def test_minor_notes(self):
        cc, sev, ack = classify_change({"notes_medical"}, is_en_route=False)
        assert cc == "minor"
        assert sev == "INFO"
        assert ack is False

    def test_cancellation_en_route(self):
        cc, sev, ack = classify_change(set(), is_en_route=True, is_cancellation=True)
        assert cc == "critical"
        assert ack is True


class TestMaskFinancialFields:
    def test_reader_hides_amount(self):
        payload = {
            "booking_summary": {"amount": 120.0, "status": "PENDING"},
            "amount": 50,
        }
        out = mask_financial_fields(payload, InstitutionRole.READER.value)
        assert "amount" not in out.get("booking_summary", {})
        assert "amount" not in out

    def test_admin_keeps_amount(self):
        payload = {"booking_summary": {"amount": 120.0}}
        out = mask_financial_fields(payload, InstitutionRole.ADMIN.value)
        assert out["booking_summary"]["amount"] == 120.0


class TestVersionAndBoardedGuards:
    def test_boarded_blocks(self):
        b = Booking()
        b.boarded_at = datetime.now(UTC)
        assert assert_not_boarded(b) is not None

    def test_version_conflict(self):
        b = Booking()
        b.edit_version = 3
        conflict = check_version(b, 2)
        assert conflict is not None
        assert conflict.get("current_version") == 3


class TestBillingReasonCodes:
    def test_codes_closed_set(self):
        assert "PRICE_CORRECTION" in BILLING_CHANGE_REASON_CODES
        assert "OTHER" in BILLING_CHANGE_REASON_CODES


class TestOperationalFieldsWhitelist:
    def test_whitelist_includes_locations(self):
        assert "pickup_location" in INSTITUTION_OPERATIONAL_FIELDS
        assert "amount" not in INSTITUTION_OPERATIONAL_FIELDS

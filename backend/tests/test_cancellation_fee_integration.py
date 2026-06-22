"""Integration tests for cancellation fee pipeline.

Tests the full flow: schema validation -> compute_cancellation_fields -> fee result.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from marshmallow import ValidationError

from application.bookings.cancellation_policy_schema import CancellationPolicySchema
from application.bookings.cancellation_rules import compute_cancellation_fields

VALID_POLICY_INPUT = {
    "enabled": True,
    "basis": "booking_amount",
    "apply_when_driver_assigned_only": True,
    "tiers": [
        {
            "id": "t1",
            "type": "time",
            "hours_before": 24,
            "percent": 20,
            "label": "< 24h",
        },
        {
            "id": "t2",
            "type": "time",
            "hours_before": 12,
            "percent": 40,
            "label": "< 12h",
        },
        {"id": "t3", "type": "time", "hours_before": 2, "percent": 60, "label": "< 2h"},
        {
            "id": "t4",
            "type": "status",
            "status": "EN_ROUTE",
            "percent": 70,
            "label": "Chauffeur en route",
        },
    ],
    "min_fee_chf": 0,
    "max_fee_chf": None,
    "reason_overrides": {
        "MAJOR_DELAY": {"billable": False},
    },
}


class TestCancellationPolicySchemaValidation:
    """Test the Marshmallow schema validates and normalizes correctly."""

    def test_valid_policy_loads(self):
        result = CancellationPolicySchema().load(VALID_POLICY_INPUT)
        assert result["enabled"] is True
        assert len(result["tiers"]) == 4

    def test_tiers_sorted_asc(self):
        """Time tiers must be sorted ASC by hours_before after load."""
        result = CancellationPolicySchema().load(VALID_POLICY_INPUT)
        time_tiers = [t for t in result["tiers"] if t["type"] == "time"]
        hours = [t["hours_before"] for t in time_tiers]
        assert hours == sorted(hours), f"Expected ascending order, got {hours}"

    def test_enabled_without_tiers_fails(self):
        with pytest.raises(ValidationError, match="tier"):
            CancellationPolicySchema().load({"enabled": True, "tiers": []})

    def test_duplicate_tier_ids_fails(self):
        bad_input = {
            **VALID_POLICY_INPUT,
            "tiers": [
                {"id": "t1", "type": "time", "hours_before": 24, "percent": 20},
                {"id": "t1", "type": "time", "hours_before": 12, "percent": 40},
            ],
        }
        with pytest.raises(ValidationError, match="unique"):
            CancellationPolicySchema().load(bad_input)

    def test_invalid_status_fails(self):
        bad_input = {
            **VALID_POLICY_INPUT,
            "tiers": [
                {"id": "t1", "type": "status", "status": "COMPLETED", "percent": 50},
            ],
        }
        with pytest.raises(ValidationError):
            CancellationPolicySchema().load(bad_input)

    def test_max_below_min_fails(self):
        bad_input = {**VALID_POLICY_INPUT, "min_fee_chf": 50, "max_fee_chf": 10}
        with pytest.raises(ValidationError, match="max_fee"):
            CancellationPolicySchema().load(bad_input)

    def test_unknown_reason_override_fails(self):
        bad_input = {
            **VALID_POLICY_INPUT,
            "reason_overrides": {"FAKE_REASON": {"billable": False}},
        }
        with pytest.raises(ValidationError, match="FAKE_REASON"):
            CancellationPolicySchema().load(bad_input)

    def test_disabled_policy_loads(self):
        result = CancellationPolicySchema().load({"enabled": False})
        assert result["enabled"] is False
        assert result["tiers"] == []


class TestComputeCancellationFieldsIntegration:
    """Test the full pipeline: compute_cancellation_fields with booking + policy."""

    def _make_booking(self, amount=100, driver_id=1, hours_ahead=10, status="ASSIGNED"):
        sched = datetime.now(UTC) + timedelta(hours=hours_ahead)
        return SimpleNamespace(
            amount=amount,
            driver_id=driver_id,
            scheduled_time=sched,
            status=status,
            company_id=1,
        )

    def test_fields_with_policy_assigned_10h(self):
        """Full pipeline: ASSIGNED, cancel 10h before, tier 12h (40%)."""
        policy = CancellationPolicySchema().load(VALID_POLICY_INPUT)
        b = self._make_booking(amount=100, hours_ahead=10)
        cancel_at = b.scheduled_time - timedelta(hours=10)

        fields = compute_cancellation_fields(
            reason_code="LAST_MINUTE",
            reason_text="Client annule",
            cancelled_by_role="company",
            now=cancel_at,
            booking=b,
            policy=policy,
            status_at_cancel="ASSIGNED",
        )

        assert fields["is_cancellation_billable"] is True
        assert fields["cancellation_fee_amount"] == Decimal("40.00")
        assert fields["cancellation_fee_percent"] == 40
        assert fields["cancellation_fee_tier_id"] == "t2"
        assert fields["cancellation_reason_code"] == "LAST_MINUTE"
        assert fields["cancelled_by_role"] == "company"

    def test_fields_en_route(self):
        """Full pipeline: EN_ROUTE -> status tier (70%)."""
        policy = CancellationPolicySchema().load(VALID_POLICY_INPUT)
        b = self._make_booking(amount=80, status="EN_ROUTE")

        fields = compute_cancellation_fields(
            reason_code="CLIENT_REQUEST",
            reason_text=None,
            cancelled_by_role="company",
            booking=b,
            policy=policy,
            status_at_cancel="EN_ROUTE",
        )

        assert fields["is_cancellation_billable"] is True
        assert fields["cancellation_fee_amount"] == Decimal("56.00")
        assert fields["cancellation_fee_percent"] == 70
        assert fields["cancellation_fee_tier_id"] == "t4"

    def test_fields_cascade_override(self):
        """Cascade from outbound -> non-billable regardless of policy."""
        policy = CancellationPolicySchema().load(VALID_POLICY_INPUT)
        b = self._make_booking()

        fields = compute_cancellation_fields(
            reason_code="OUTBOUND_CANCELLED",
            reason_text="Retour annulé automatiquement",
            cancelled_by_role="system",
            booking=b,
            policy=policy,
            cancel_source="cascade_from_outbound",
            status_at_cancel="ASSIGNED",
        )

        assert fields["is_cancellation_billable"] is False
        assert fields["cancellation_fee_amount"] == Decimal("0")

    def test_fields_no_policy(self):
        """No policy -> legacy reason-based, fee fields populated with 0."""
        b = self._make_booking()

        fields = compute_cancellation_fields(
            reason_code="LAST_MINUTE",
            reason_text=None,
            cancelled_by_role="company",
            booking=b,
            policy=None,
            status_at_cancel="ASSIGNED",
        )

        assert fields["is_cancellation_billable"] is True
        assert fields["cancellation_fee_amount"] == Decimal("0")
        assert fields["cancellation_fee_percent"] is None
        assert fields["cancellation_fee_tier_id"] is None

    def test_fields_client_request_no_policy_early_cancel(self):
        """CLIENT_REQUEST sans politique, annulation anticipée -> non facturable."""
        b = self._make_booking(hours_ahead=120)

        fields = compute_cancellation_fields(
            reason_code="CLIENT_REQUEST",
            reason_text="Client a demandé l'annulation",
            cancelled_by_role="institution",
            booking=b,
            policy=None,
            status_at_cancel="ACCEPTED",
        )

        assert fields["is_cancellation_billable"] is False
        assert fields["cancellation_fee_amount"] == Decimal("0")
        """MAJOR_DELAY with override billable=false -> non-billable."""
        policy = CancellationPolicySchema().load(VALID_POLICY_INPUT)
        b = self._make_booking()

        fields = compute_cancellation_fields(
            reason_code="MAJOR_DELAY",
            reason_text="Gros retard",
            cancelled_by_role="company",
            booking=b,
            policy=policy,
            status_at_cancel="ASSIGNED",
        )

        assert fields["is_cancellation_billable"] is False
        assert fields["cancellation_fee_amount"] == Decimal("0")

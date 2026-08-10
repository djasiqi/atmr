"""Tests unitaires — règles d'investigation admin transports."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from models import BookingStatus
from services.admin_booking_investigation import (
    build_investigation_reasons,
    build_support_diagnostic,
    compute_needs_investigation_booking,
    evaluate_incomplete,
    has_blocking_reason,
)
from services.admin_booking_support_detail import (
    _sanitize_details,
    serialize_admin_support_transport,
)


def _booking(**kwargs):
    defaults = {
        "scheduled_time": datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
        "customer_name": "Sofia",
        "pickup_location": "HUG",
        "dropoff_location": "Chemin 9",
        "status": BookingStatus.ACCEPTED,
        "driver_id": None,
        "client": None,
        "id": 3,
        "amount": 50.0,
        "mission_type": "patient_transport",
        "is_round_trip": False,
        "is_return": False,
        "created_at": datetime(2026, 7, 31, tzinfo=UTC),
        "updated_at": datetime(2026, 7, 31, tzinfo=UTC),
        "cancelled_at": None,
        "edit_version": 1,
        "customer_full_name": "Sofia GIUSEPPA",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestInvestigationReasons:
    def test_missing_scheduled_time_blocking(self):
        b = _booking(scheduled_time=None)
        reasons = build_investigation_reasons(
            b, created_by={"source": "client", "label": "X"}, has_pending_transfer=False
        )
        codes = [r["code"] for r in reasons]
        assert "MISSING_SCHEDULED_TIME" in codes
        assert has_blocking_reason(reasons)
        assert compute_needs_investigation_booking(
            b, has_pending_transfer=False, created_by={"source": "client"}
        )

    def test_private_booking_no_missing_institution(self):
        b = _booking()
        reasons = build_investigation_reasons(
            b,
            created_by={"source": "client", "label": "Client"},
            has_pending_transfer=False,
            institution_present=False,
        )
        assert all(r["code"] != "MISSING_INSTITUTION" for r in reasons)

    def test_institution_request_without_institution_warning(self):
        b = _booking()
        reasons = build_investigation_reasons(
            b,
            created_by={"source": "institution_request", "label": "Nurse"},
            has_pending_transfer=False,
            institution_present=False,
        )
        warning = next(r for r in reasons if r["code"] == "MISSING_INSTITUTION")
        assert warning["severity"] == "warning"
        assert not has_blocking_reason(
            [r for r in reasons if r["code"] == "MISSING_INSTITUTION"]
        )

    def test_unknown_creator_is_info(self):
        b = _booking()
        reasons = build_investigation_reasons(
            b,
            created_by={"source": "unknown", "label": None},
            has_pending_transfer=False,
        )
        info = next(r for r in reasons if r["code"] == "MISSING_CREATOR")
        assert info["severity"] == "info"
        diag = build_support_diagnostic(reasons, status_label="Acceptée")
        assert diag["status"] == "ok" or diag["needs_investigation"] is False
        # info alone → ok
        assert diag["status"] == "ok"

    def test_accepted_without_driver_no_driver_reason(self):
        b = _booking(status=BookingStatus.ACCEPTED, driver_id=None)
        reasons = build_investigation_reasons(
            b, created_by={"source": "client", "label": "X"}, has_pending_transfer=False
        )
        assert all(r["code"] != "DRIVER_INVARIANT_BROKEN" for r in reasons)

    def test_assigned_without_driver_blocking_unit(self):
        """Défense en profondeur hors contrainte PG."""
        b = _booking(status=BookingStatus.ASSIGNED, driver_id=None)
        reasons = build_investigation_reasons(
            b, created_by={"source": "client", "label": "X"}, has_pending_transfer=False
        )
        assert any(r["code"] == "DRIVER_INVARIANT_BROKEN" for r in reasons)
        assert has_blocking_reason(reasons)

    def test_past_due_pending_code(self):
        now = datetime(2026, 8, 3, tzinfo=UTC)
        b = _booking(
            status=BookingStatus.PENDING,
            scheduled_time=now - timedelta(hours=48),
        )
        reasons = build_investigation_reasons(
            b,
            created_by={"source": "client", "label": "X"},
            has_pending_transfer=False,
            now=now,
        )
        assert any(r["code"] == "PAST_DUE_PENDING_24H" for r in reasons)

    def test_pending_transfer_code(self):
        b = _booking()
        reasons = build_investigation_reasons(
            b, created_by={"source": "client", "label": "X"}, has_pending_transfer=True
        )
        assert any(r["code"] == "PENDING_TRANSFER_REQUIRES_REVIEW" for r in reasons)

    def test_diagnostic_three_states(self):
        blocking = [
            {
                "code": "MISSING_SCHEDULED_TIME",
                "severity": "blocking",
                "label": "Horaire manquant",
                "recommended_action": "request_or_correct_schedule",
            }
        ]
        d1 = build_support_diagnostic(
            blocking, status_label="Acceptée", current_company_name="Diaz"
        )
        assert d1["status"] == "action_required"
        assert d1["needs_investigation"] is True
        assert "Diaz" in d1["summary"]

        warning = [
            {
                "code": "MISSING_INSTITUTION",
                "severity": "warning",
                "label": "Institution absente",
                "recommended_action": None,
            }
        ]
        d2 = build_support_diagnostic(warning, status_label="Acceptée")
        assert d2["status"] == "attention"
        assert d2["needs_investigation"] is False

        info_only = [
            {
                "code": "MISSING_CREATOR",
                "severity": "info",
                "label": "Auteur inconnu",
                "recommended_action": None,
            }
        ]
        d3 = build_support_diagnostic(info_only)
        assert d3["status"] == "ok"


class TestSupportTransportDto:
    def test_serialize_excludes_pii_keys(self):
        b = _booking()
        dto = serialize_admin_support_transport(b)
        forbidden = {
            "birth_date",
            "notes_medical",
            "door_code",
            "pickup_door_code",
            "contact_phone",
            "pickup_lat",
            "billing",
            "online_payment",
        }
        assert forbidden.isdisjoint(dto.keys())
        assert "amount_chf" in dto
        assert "scheduled_at" in dto
        assert "last_updated_age_seconds" in dto

    def test_sanitize_strips_secrets_and_unknown_keys(self):
        raw = {
            "from_status": "PENDING",
            "to_status": "ACCEPTED",
            "token": "secret-value",
            "password": "x",
            "extra_internal": "nope",
        }
        filtered = _sanitize_details(
            raw, allowed_keys={"from_status", "to_status", "reason"}
        )
        assert filtered == {"from_status": "PENDING", "to_status": "ACCEPTED"}


class TestEvaluateIncomplete:
    def test_incomplete_parity_fields(self):
        assert evaluate_incomplete(_booking(scheduled_time=None)) is True
        assert evaluate_incomplete(_booking(customer_name="")) is True
        assert evaluate_incomplete(_booking()) is False

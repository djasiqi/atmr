"""Tests service audit / versioning bookings institution."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from application.institutions.destination_details_rules import (
    MEDICAL_DESTINATION_OR_ERROR,
)
from models import Booking, Institution, TransportRequest, User, UserRole
from models.enums import BookingStatus, InstitutionRole, RequestStatus
from services.institutions.booking_change_service import (
    BILLING_CHANGE_REASON_CODES,
    INSTITUTION_OPERATIONAL_FIELDS,
    LEG_SCHEDULE_PATCH_FIELDS,
    InstitutionBookingContext,
    _hhmm_label,
    _medical_destination_patch_error,
    _strip_noop_operational_patch,
    _wall_clock_key,
    assert_not_boarded,
    check_version,
    classify_change,
    compute_appointment_diff,
    mask_financial_fields,
    split_destination_and_return_legs,
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
        cc, _sev, ack = classify_change(set(), is_en_route=True, is_cancellation=True)
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


class TestLegSchedulePatchFields:
    def test_leg_schedule_fields_allowed(self):
        assert "appointment_time" in LEG_SCHEDULE_PATCH_FIELDS
        assert "leg_appointments" in LEG_SCHEDULE_PATCH_FIELDS
        assert "return_appointment_time" in LEG_SCHEDULE_PATCH_FIELDS


class TestAppointmentDiff:
    def test_detects_nested_rdv_change(self):
        diff = compute_appointment_diff(
            dest_keys=["2026-09-12T14:00"],
            return_key=None,
            payload={
                "leg_appointments": [
                    {"index": 0, "scheduled_time": "2026-09-12T13:00:00"},
                ]
            },
        )
        assert diff["changed_dest_indices"] == [0]
        assert diff["after_dest"][0] == "2026-09-12T13:00"
        assert diff["return_changed"] is False

    def test_detects_later_rdv_same_rule(self):
        diff = compute_appointment_diff(
            dest_keys=["2026-09-12T14:00"],
            return_key=None,
            payload={"appointment_time": "2026-09-12T15:00:00"},
        )
        assert diff["changed_dest_indices"] == [0]
        assert diff["after_dest"][0] == "2026-09-12T15:00"

    def test_no_patch_is_unchanged(self):
        diff = compute_appointment_diff(
            dest_keys=["2026-09-12T14:00"],
            return_key=None,
            payload={"pickup_access_notes": "Contact: Admin"},
        )
        assert diff["changed_dest_indices"] == []
        assert diff["return_changed"] is False

    def test_same_rdv_is_unchanged(self):
        diff = compute_appointment_diff(
            dest_keys=["2026-09-12T14:00"],
            return_key=None,
            payload={
                "leg_appointments": [
                    {"index": 0, "scheduled_time": "2026-09-12T14:00:00"},
                ]
            },
        )
        assert diff["changed_dest_indices"] == []

    def test_return_split_keeps_dest_indices(self):
        dest, ret = split_destination_and_return_legs(
            ["A", "B", "C"],
            return_to_institution=True,
        )
        assert dest == ["A", "B"]
        assert ret == "C"

    def test_wall_clock_and_label(self):
        assert _wall_clock_key("2026-09-12T14:00:00") == "2026-09-12T14:00"
        assert _hhmm_label("2026-09-12T14:00") == "14:00"

    def test_fat_form_payload_empty_doctor_is_noop(self):
        booking = Booking()
        booking.customer_name = "Charlotte CAVADINI"
        booking.pickup_location = "Chemin des Courbes 9, 1247, Anières"
        booking.dropoff_location = "HUG"
        booking.scheduled_time = datetime(2026, 9, 12, 13, 15)
        booking.hospital_service = "Radiologie"
        booking.doctor_name = None
        booking.wheelchair_need = False
        booking.wheelchair_client_has = False
        cleaned = _strip_noop_operational_patch(
            booking,
            {
                "pickup_location": "Chemin des Courbes 9, 1247, Anières",
                "dropoff_location": "HUG",
                "scheduled_time": "2026-09-12T13:15:00",
                "hospital_service": "Radiologie",
                "doctor_name": "",
                "wheelchair_need": False,
            },
        )
        assert cleaned == {}


def _medical_ctx(
    *,
    destination_type: str | None = None,
    dropoff_type: str | None = "other",
    mission_type: str = "patient_transport",
    hospital_service: str | None = None,
    doctor_name: str | None = None,
    routing: dict | None = None,
) -> InstitutionBookingContext:
    billing_details = {"routing": routing} if routing is not None else {}
    return InstitutionBookingContext(
        booking=SimpleNamespace(
            hospital_service=hospital_service,
            doctor_name=doctor_name,
            mission_type=mission_type,
        ),
        transport_request=SimpleNamespace(
            dropoff_type=dropoff_type,
            destination_type=destination_type,
            mission_type=mission_type,
            billing_details=billing_details,
        ),
        institution_id=1,
    )


class TestMedicalDestinationPatch:
    def test_rejects_both_empty_on_explicit_medical(self):
        err = _medical_destination_patch_error(
            _medical_ctx(),
            {
                "destination_type": "medical",
                "hospital_service": "",
                "doctor_name": "",
            },
        )
        assert err is not None
        assert err["error"] == "Données invalides"
        assert MEDICAL_DESTINATION_OR_ERROR in str(err["details"])

    def test_other_allows_both_empty(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(),
                {
                    "destination_type": "other",
                    "hospital_service": "",
                    "doctor_name": "",
                },
            )
            is None
        )

    def test_legacy_other_dropoff_type_is_not_medical(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(dropoff_type="other"),
                {"hospital_service": "", "doctor_name": ""},
            )
            is None
        )

    def test_persisted_routing_medical_rejects_both_empty(self):
        err = _medical_destination_patch_error(
            _medical_ctx(routing={"destination_type": "medical"}),
            {"hospital_service": "", "doctor_name": ""},
        )
        assert err is not None

    def test_accepts_service_only(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(),
                {
                    "destination_type": "medical",
                    "hospital_service": "Radiologie",
                    "doctor_name": "",
                },
            )
            is None
        )

    def test_accepts_doctor_only(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(),
                {
                    "destination_type": "medical",
                    "hospital_service": "",
                    "doctor_name": "Dr Martin",
                },
            )
            is None
        )

    def test_accepts_both(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(),
                {
                    "destination_type": "medical",
                    "hospital_service": "Cardiologie",
                    "doctor_name": "Dr Martin",
                },
            )
            is None
        )

    def test_notes_only_skips_even_if_booking_incomplete(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(),
                {"notes_medical": "RAS"},
            )
            is None
        )

    def test_partial_clear_service_keeps_existing_doctor(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(doctor_name="Dr Martin"),
                {"hospital_service": ""},
            )
            is None
        )

    def test_domicile_allows_both_empty(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(dropoff_type="domicile"),
                {"hospital_service": "", "doctor_name": ""},
            )
            is None
        )

    def test_material_delivery_allows_both_empty(self):
        assert (
            _medical_destination_patch_error(
                _medical_ctx(mission_type="material_delivery"),
                {
                    "hospital_service": "",
                    "doctor_name": "",
                    "mission_type": "material_delivery",
                },
            )
            is None
        )

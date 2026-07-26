# ruff: noqa: I001
"""Tests: création du booking retour lors de l'acceptation d'une demande A/R.

Couvre les sections 1-7 du plan (validation, création retour, inversion
pickup/dropoff, billing, booking_summary agrégé, chat unifié, résolution
TransportRequest pour retour).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from application.institutions.accept_offer import (
    AcceptOfferInput,
    AcceptOfferUseCase,
    _should_use_legs_conversion,
)
from schemas.institution_schemas import TransportRequestCreateSchema


def _future_mission_date_str(*, days_ahead: int = 10) -> str:
    """Date mission future (évite validation « dans le passé »)."""
    return (date.today() + timedelta(days=days_ahead)).isoformat()


def _future_scheduled_iso(
    *, days_ahead: int = 10, hour: int = 10, minute: int = 0
) -> str:
    """ISO8601 futur avec offset +01:00 pour tests schema."""
    mission = date.today() + timedelta(days=days_ahead)
    return f"{mission.isoformat()}T{hour:02d}:{minute:02d}:00+01:00"


# ── Lightweight stubs (pas de DB réelle) ──


@dataclass
class _Institution:
    id: int = 1
    name: str = "Clinique Test"


@dataclass
class _Patient:
    id: int = 1
    first_name: str = "Jean"
    last_name: str = "Dupont"
    dob: date | None = date(1985, 3, 15)


@dataclass
class _Leg:
    sequence_index: int = 0
    route_sequence_number: int = 1
    pickup_location: str = "Domicile"
    dropoff_location: str = "HUG"
    dropoff_establishment: str | None = "HUG"
    dropoff_service: str | None = "Neurologie"
    dropoff_doctor: str | None = "Dr Test"
    scheduled_time: datetime | None = None
    time_confirmed: bool = False
    pickup_lat: float | None = None
    pickup_lng: float | None = None
    dropoff_lat: float | None = None
    dropoff_lng: float | None = None
    is_return_stop: bool = False
    booking_id: int | None = None


@dataclass
class _Client:
    id: int = 10
    preferential_rate: Decimal | None = Decimal("50.00")
    default_billed_to_type: str = "clinic"
    institution_name: str | None = "Clinique Test"
    linked_institution_id: int | None = None


@dataclass
class _Booking:
    id: int = 0
    company_id: int = 0
    user_id: int = 0
    client_id: int | None = None
    customer_name: str = ""
    mission_type: str = "patient_transport"
    delivery_description: str | None = None
    scheduled_time: datetime | None = None
    is_round_trip: bool = False
    is_return: bool = False
    parent_booking_id: int | None = None
    pickup_location: str = ""
    pickup_lat: float | None = None
    pickup_lon: float | None = None
    pickup_access_notes: str | None = None
    pickup_floor: str | None = None
    pickup_door_code: str | None = None
    dropoff_location: str = ""
    dropoff_lat: float | None = None
    dropoff_lon: float | None = None
    dropoff_access_notes: str | None = None
    dropoff_floor: str | None = None
    dropoff_door_code: str | None = None
    medical_facility: str | None = None
    hospital_service: str | None = None
    doctor_name: str | None = None
    wheelchair_client_has: bool = False
    wheelchair_need: bool = False
    notes_medical: str | None = None
    billed_to_type: str = "patient"
    billed_to_company_id: int | None = None
    billing_party_id: int | None = None
    status: str = "PENDING"
    amount: Decimal = Decimal("50.00")
    return_trip: _Booking | None = None
    invoice_line_id: int | None = None
    time_confirmed: bool = True


@dataclass
class _TransportRequest:
    id: int = 100
    institution_id: int = 1
    institution: _Institution | None = field(default_factory=_Institution)
    patient: _Patient | None = field(default_factory=_Patient)
    pickup_location: str = "10 Rue de Test"
    pickup_lat: float | None = 46.2
    pickup_lng: float | None = 6.1
    pickup_floor: str | None = None
    pickup_door_code: str | None = None
    pickup_entry_point: str | None = None
    dropoff_location: str = "Hôpital Central"
    dropoff_lat: float | None = 46.3
    dropoff_lng: float | None = 6.2
    dropoff_floor: str | None = None
    dropoff_door_code: str | None = None
    dropoff_entry_point: str | None = None
    floor_elevator_info: str | None = None
    contact_on_site: dict[str, Any] | None = None
    mission_type: str = "patient_transport"
    delivery_description: str | None = None
    mobility: dict[str, bool] | None = None
    notes: str | None = "Notes test"
    billing_intent: str = "patient"
    billing_details: dict[str, Any] | None = None
    is_round_trip: bool = False
    return_time: datetime | None = None
    return_date: date | None = None
    return_time_confirmed: bool = False
    scheduled_time: datetime = field(
        default_factory=lambda: datetime.now(UTC) + timedelta(hours=2)
    )
    booking_id: int | None = None
    public_id: str = "PUB-123"
    external_reference: str = "EXT-001"
    pickup_type: str | None = None
    dropoff_type: str | None = None
    legs: list | None = None
    status: str = "SENT"
    multi_stop: bool = False
    return_to_institution: bool = False


# ── Schema validation tests ──


class TestTransportRequestSchemaRoundTrip:
    """Validation schema : A/R avec return_date ou return_time."""

    def test_schema_rejects_round_trip_without_return_plan(self):
        schema = TransportRequestCreateSchema()
        _future_mission_date_str()
        data = {
            "external_reference": "REF-001",
            "scheduled_time": _future_scheduled_iso(),
            "pickup_location": "A",
            "dropoff_location": "B",
            "is_round_trip": True,
        }
        from marshmallow import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            schema.load(data)
        assert "return_date" in str(exc_info.value) or "return_time" in str(
            exc_info.value
        )

    def test_schema_accepts_round_trip_with_return_date_only(self):
        schema = TransportRequestCreateSchema()
        mission = _future_mission_date_str()
        data = {
            "external_reference": "REF-001B",
            "scheduled_time": _future_scheduled_iso(),
            "pickup_location": "A",
            "dropoff_location": "B",
            "is_round_trip": True,
            "return_date": mission,
        }
        result = schema.load(data)
        assert result["is_round_trip"] is True
        assert result["return_date"] == mission
        assert result.get("return_time") is None

    def test_schema_accepts_round_trip_with_return_time(self):
        schema = TransportRequestCreateSchema()
        _future_mission_date_str()
        data = {
            "external_reference": "REF-002",
            "scheduled_time": _future_scheduled_iso(),
            "pickup_location": "A",
            "dropoff_location": "B",
            "is_round_trip": True,
            "return_time": _future_scheduled_iso(hour=16, minute=0),
        }
        result = schema.load(data)
        assert result["is_round_trip"] is True
        assert result["return_time"] is not None

    def test_schema_accepts_simple_trip(self):
        schema = TransportRequestCreateSchema()
        data = {
            "external_reference": "REF-003",
            "scheduled_time": _future_scheduled_iso(),
            "pickup_location": "A",
            "dropoff_location": "B",
            "is_round_trip": False,
        }
        result = schema.load(data)
        assert result["is_round_trip"] is False

    def test_schema_accepts_simple_trip_without_external_reference(self):
        schema = TransportRequestCreateSchema()
        data = {
            "scheduled_time": _future_scheduled_iso(),
            "pickup_location": "A",
            "dropoff_location": "B",
            "is_round_trip": False,
        }
        result = schema.load(data)
        assert result["is_round_trip"] is False
        assert "external_reference" not in result


# ── Use-case unit tests (mocked DB) ──


class TestAcceptOfferRoundTrip:
    """Tests unitaires pour _create_booking_from_request avec A/R."""

    def _make_uc(self) -> AcceptOfferUseCase:
        return AcceptOfferUseCase()

    @patch("application.institutions.accept_offer.Client")
    def test_get_or_create_institution_client_fallback_by_name_keeps_rate(
        self, mock_client_cls: MagicMock
    ):
        """Si le lien FK manque, fallback par nom réutilise le client tarifé."""
        uc = self._make_uc()
        tr = _TransportRequest(
            institution=_Institution(id=42, name="Clinique Les Hauts d’Anières")
        )
        existing_client = _Client(
            id=99,
            preferential_rate=Decimal("88.00"),
            institution_name="Clinique Les Hauts d'Anieres",
            linked_institution_id=None,
        )

        fk_lookup = MagicMock()
        fk_lookup.first.return_value = None
        name_lookup = MagicMock()
        name_lookup.all.return_value = [existing_client]
        mock_client_cls.query.filter.side_effect = [fk_lookup, name_lookup]

        uc._create_institution_client = MagicMock(return_value=None)  # type: ignore[assignment]

        resolved = uc._get_or_create_institution_client(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
        )

        assert resolved is existing_client
        assert resolved.preferential_rate == Decimal("88.00")
        assert resolved.linked_institution_id == 42
        uc._create_institution_client.assert_not_called()

    @patch("application.institutions.accept_offer.db")
    def test_round_trip_creates_two_bookings(self, mock_db: MagicMock):
        """A/R avec return_time → 2 bookings (aller + retour)."""
        uc = self._make_uc()
        uc._get_or_create_institution_client = MagicMock(return_value=_Client())  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        return_dt = datetime.now(UTC) + timedelta(hours=6)
        tr = _TransportRequest(is_round_trip=True, return_time=return_dt)

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if b.id == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert outbound is not None
        assert return_booking is not None
        assert outbound.is_round_trip is True
        assert outbound.is_return is not True
        assert return_booking.is_return is True
        assert return_booking.is_round_trip is False
        assert return_booking.parent_booking_id == outbound.id
        from shared.time_utils import api_scheduled_iso_to_naive_geneva

        expected_naive = api_scheduled_iso_to_naive_geneva(return_dt)
        actual_naive = return_booking.scheduled_time
        assert actual_naive == expected_naive

    @patch("application.institutions.accept_offer.db")
    def test_round_trip_inverts_pickup_dropoff(self, mock_db: MagicMock):
        """Retour inverse pickup ↔ dropoff (y compris access_notes)."""
        uc = self._make_uc()
        uc._get_or_create_institution_client = MagicMock(return_value=_Client())  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        return_dt = datetime.now(UTC) + timedelta(hours=6)
        tr = _TransportRequest(
            is_round_trip=True,
            return_time=return_dt,
            pickup_location="Domicile Patient",
            pickup_lat=46.2,
            pickup_lng=6.1,
            dropoff_location="Hôpital Central",
            dropoff_lat=46.3,
            dropoff_lng=6.2,
        )

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if b.id == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert return_booking is not None
        assert return_booking.pickup_location == outbound.dropoff_location
        assert return_booking.dropoff_location == outbound.pickup_location
        assert return_booking.pickup_lat == outbound.dropoff_lat
        assert return_booking.pickup_lon == outbound.dropoff_lon
        assert return_booking.dropoff_lat == outbound.pickup_lat
        assert return_booking.dropoff_lon == outbound.pickup_lon
        assert return_booking.pickup_access_notes == outbound.dropoff_access_notes
        assert return_booking.dropoff_access_notes == outbound.pickup_access_notes
        assert return_booking.pickup_floor == outbound.dropoff_floor
        assert return_booking.pickup_door_code == outbound.dropoff_door_code
        assert return_booking.dropoff_floor == outbound.pickup_floor
        assert return_booking.dropoff_door_code == outbound.pickup_door_code

    @patch("application.institutions.accept_offer.db")
    def test_simple_trip_creates_one_booking(self, mock_db: MagicMock):
        """Trajet simple (pas A/R) → 1 seul booking, pas de retour."""
        uc = self._make_uc()
        uc._get_or_create_institution_client = MagicMock(return_value=_Client())  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        tr = _TransportRequest(is_round_trip=False, return_time=None)

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if b.id == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert outbound is not None
        assert return_booking is None

    @patch("application.institutions.accept_offer.db")
    def test_round_trip_same_billing(self, mock_db: MagicMock):
        """A/R: amount, billed_to_type, billed_to_company_id identiques."""
        uc = self._make_uc()
        client = _Client(
            preferential_rate=Decimal("75.00"), default_billed_to_type="clinic"
        )
        uc._get_or_create_institution_client = MagicMock(return_value=client)  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        return_dt = datetime.now(UTC) + timedelta(hours=6)
        tr = _TransportRequest(is_round_trip=True, return_time=return_dt)

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if b.id == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert return_booking is not None
        assert return_booking.amount == outbound.amount
        assert return_booking.billed_to_type == outbound.billed_to_type
        assert return_booking.billed_to_company_id == outbound.billed_to_company_id

    @patch("application.institutions.accept_offer.db")
    def test_billing_intent_institution_sets_clinic_and_company(
        self, mock_db: MagicMock
    ):
        """billing_intent=institution -> billed_to_type=clinic, company_id accepté."""
        uc = self._make_uc()
        client = _Client(
            preferential_rate=Decimal("75.00"), default_billed_to_type="clinic"
        )
        uc._get_or_create_institution_client = MagicMock(return_value=client)  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]
        uc._resolve_billed_to_company_id_before_flush = MagicMock(return_value=None)  # type: ignore[method-assign]

        tr = _TransportRequest(
            is_round_trip=False,
            return_time=None,
            billing_intent="institution",
        )

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if getattr(b, "id", 0) == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, _return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert outbound.billed_to_type == "clinic"
        assert outbound.billed_to_company_id is None

    @patch("application.institutions.accept_offer.db")
    def test_billing_intent_patient_keeps_patient_billing(self, mock_db: MagicMock):
        """billing_intent=patient -> billed_to_type=patient sans company_id."""
        uc = self._make_uc()
        client = _Client(
            preferential_rate=Decimal("75.00"), default_billed_to_type="clinic"
        )
        uc._get_or_create_institution_client = MagicMock(return_value=client)  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        tr = _TransportRequest(
            is_round_trip=False,
            return_time=None,
            billing_intent="patient",
        )
        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if getattr(b, "id", 0) == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, _return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert outbound.billed_to_type == "patient"
        assert outbound.billed_to_company_id is None

    @patch("application.institutions.accept_offer.db")
    def test_round_trip_return_date_only_null_time_not_confirmed(
        self, mock_db: MagicMock
    ):
        """return_date seul → scheduled_time=null, time_confirmed=false."""
        uc = self._make_uc()
        uc._get_or_create_institution_client = MagicMock(return_value=_Client())  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        tr = _TransportRequest(
            is_round_trip=True,
            return_date=date(2026, 5, 25),
            return_time=None,
            return_time_confirmed=False,
        )

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if getattr(b, "id", 0) == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        _outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert return_booking is not None
        assert return_booking.time_confirmed is False
        assert return_booking.scheduled_time is None

    @patch("application.institutions.accept_offer.db")
    def test_round_trip_real_midnight_confirmed(self, mock_db: MagicMock):
        """Minuit réel confirmé → scheduled_time=00:00, time_confirmed=true."""
        uc = self._make_uc()
        uc._get_or_create_institution_client = MagicMock(return_value=_Client())  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        return_dt = datetime(2026, 5, 25, 0, 0, 0)
        tr = _TransportRequest(
            is_round_trip=True,
            return_time=return_dt,
            return_time_confirmed=True,
        )

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if getattr(b, "id", 0) == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        _outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert return_booking is not None
        assert return_booking.time_confirmed is True
        assert return_booking.scheduled_time.hour == 0
        assert return_booking.scheduled_time.minute == 0


class TestMultiStopReturnConversion:
    """A/R multi-étapes : return_to_institution + legs sans is_round_trip."""

    def test_should_use_legs_conversion_with_two_legs_without_flag(self):
        tr = _TransportRequest(
            multi_stop=False,
            return_to_institution=True,
            legs=[
                _Leg(
                    sequence_index=0, pickup_location="Clinique", dropoff_location="HUG"
                ),
                _Leg(
                    sequence_index=1,
                    pickup_location="HUG",
                    dropoff_location="Clinique",
                    is_return_stop=True,
                ),
            ],
        )
        assert _should_use_legs_conversion(tr) is True

    def test_should_use_legs_conversion_false_for_simple_trip(self):
        tr = _TransportRequest(is_round_trip=False, legs=[])
        assert _should_use_legs_conversion(tr) is False

    @patch("application.institutions.accept_offer.db")
    def test_institution_return_without_is_round_trip(self, mock_db: MagicMock):
        """return_to_institution + leg retour → booking retour + legs liés."""
        uc = AcceptOfferUseCase()
        uc._get_or_create_institution_client = MagicMock(return_value=_Client())  # type: ignore[assignment]
        uc._resolve_billing_party = MagicMock()  # type: ignore[assignment]

        return_leg = _Leg(
            sequence_index=1,
            pickup_location="HUG",
            dropoff_location="Courbes 9",
            is_return_stop=True,
            scheduled_time=None,
            time_confirmed=False,
        )
        outbound_leg = _Leg(
            sequence_index=0,
            pickup_location="Courbes 9",
            dropoff_location="HUG",
        )
        tr = _TransportRequest(
            is_round_trip=False,
            multi_stop=False,
            return_to_institution=True,
            return_date=date(2026, 7, 10),
            return_time=None,
            pickup_location="Courbes 9",
            dropoff_location="HUG",
            legs=[outbound_leg, return_leg],
        )

        _next_id = iter(range(1, 100))

        def _flush_side_effect():
            for call_args in mock_db.session.add.call_args_list:
                b = call_args[0][0]
                if getattr(b, "id", 0) == 0:
                    b.id = next(_next_id)

        mock_db.session.flush.side_effect = _flush_side_effect

        outbound, return_booking = uc._create_booking_from_request(
            transport_request=tr,  # type: ignore[arg-type]
            company_id=5,
            user_id=1,
        )

        assert outbound is not None
        assert return_booking is not None
        assert return_booking.is_return is True
        assert return_booking.parent_booking_id == outbound.id
        assert return_booking.pickup_location == "HUG"
        assert return_booking.dropoff_location == "Courbes 9"
        assert return_booking.scheduled_time is None
        assert return_booking.time_confirmed is False
        assert outbound_leg.booking_id == outbound.id
        assert return_leg.booking_id == return_booking.id


# ── booking_summary aggregation ──


class TestBookingSummaryRoundTrip:
    """booking_summary agrège les données A/R (R1 + R3)."""

    def test_booking_summary_aggregates_round_trip(self):
        from models.transport_request import _compute_overall_status, _status_value

        outbound = _Booking(
            id=1,
            status="COMPLETED",
            amount=Decimal("50.00"),
            scheduled_time=datetime.now(UTC),
        )
        ret = _Booking(
            id=2,
            status="PENDING",
            amount=Decimal("50.00"),
            is_return=True,
            parent_booking_id=1,
            scheduled_time=datetime.now(UTC) + timedelta(hours=4),
        )
        outbound.return_trip = ret

        overall = _compute_overall_status(
            _status_value(outbound.status),
            _status_value(ret.status),
        )
        assert overall == "outbound_completed"

        total = float(outbound.amount) + float(ret.amount)
        assert total == 100.0

    def test_overall_status_cancelled(self):
        from models.transport_request import _compute_overall_status

        assert _compute_overall_status("CANCELED", "PENDING") == "cancelled"
        assert _compute_overall_status("COMPLETED", "CANCELED") == "cancelled"

    def test_overall_status_completed(self):
        from models.transport_request import _compute_overall_status

        assert _compute_overall_status("COMPLETED", "COMPLETED") == "completed"
        assert _compute_overall_status("COMPLETED", "RETURN_COMPLETED") == "completed"

    def test_overall_status_in_progress(self):
        from models.transport_request import _compute_overall_status

        assert _compute_overall_status("EN_ROUTE", "PENDING") == "in_progress"
        assert _compute_overall_status("IN_PROGRESS", "PENDING") == "in_progress"

    def test_overall_status_planned(self):
        from models.transport_request import _compute_overall_status

        assert _compute_overall_status("PENDING", "PENDING") == "planned"


# ── get_request_info_from_booking for return bookings ──


class TestGetRequestInfoReturnBooking:
    """get_request_info_from_booking résout via parent_booking_id pour retours."""

    def test_resolves_return_booking_via_parent(self):
        mock_transport_req = MagicMock()
        mock_transport_req.id = 100
        mock_transport_req.public_id = "PUB-100"
        mock_transport_req.institution_id = 1
        mock_transport_req.external_reference = "EXT-001"

        mock_return_booking = MagicMock()
        mock_return_booking.is_return = True
        mock_return_booking.parent_booking_id = 42

        call_count = [0]

        def _filter_by_side_effect(**kwargs):
            call_count[0] += 1
            mock_first = MagicMock()
            if call_count[0] == 1:
                mock_first.first.return_value = None
            else:
                mock_first.first.return_value = mock_transport_req
            return mock_first

        mock_tr_cls = MagicMock()
        mock_tr_cls.query.filter_by.side_effect = _filter_by_side_effect

        mock_booking_cls = MagicMock()

        mock_db = MagicMock()
        mock_db.session.get.return_value = mock_return_booking

        with (
            patch.dict("sys.modules", {}),
            patch("models.TransportRequest", mock_tr_cls, create=True),
            patch("models.Booking", mock_booking_cls, create=True),
            patch("ext.db", mock_db),
        ):
            from importlib import reload
            import services.events.institution_events as mod

            reload(mod)
            result = mod.get_request_info_from_booking(99)

        assert result is not None
        assert result["request_id"] == 100
        assert result["institution_id"] == 1


class TestAcceptOfferClinicalDropoff:
    """Copie établissement / service / médecin depuis les legs institution."""

    def test_apply_clinical_dropoff_fields(self):
        booking = _Booking()
        AcceptOfferUseCase._apply_clinical_dropoff_fields(
            booking,
            establishment="HUG",
            service="Neurologie",
            doctor="Dr MOLLIQAJ",
        )
        assert booking.medical_facility == "HUG"
        assert booking.hospital_service == "Neurologie"
        assert booking.doctor_name == "Dr MOLLIQAJ"

    def test_clinical_dropoff_from_request_uses_first_leg(self):
        uc = AcceptOfferUseCase()
        tr = _TransportRequest(
            legs=[
                _Leg(
                    sequence_index=0,
                    dropoff_establishment="HUG",
                    dropoff_service="Neurologie",
                    dropoff_doctor="Dr MOLLIQAJ",
                ),
                _Leg(sequence_index=1, dropoff_establishment="Retour"),
            ],
        )
        est, svc, doc = uc._clinical_dropoff_from_request(tr)
        assert est == "HUG"
        assert svc == "Neurologie"
        assert doc == "Dr MOLLIQAJ"

    def test_build_metadata_uses_patient_dob(self):
        uc = AcceptOfferUseCase()
        tr = _TransportRequest(patient=_Patient(dob=date(1990, 5, 12)))
        meta = uc._build_metadata(tr)
        assert meta["patient"]["date_of_birth"] == "1990-05-12"

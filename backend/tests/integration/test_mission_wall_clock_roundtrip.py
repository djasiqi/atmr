# ruff: noqa: I001
"""STOP GATE P2.5 — round-trip heure murale mission institution (Cas 1-4).

Prouve que les flux métier réels préservent l'heure murale Genève (ex. 12:30)
de la saisie jusqu'à la projection institution / entreprise.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

import pytest

from application.institutions.accept_offer import (
    AcceptOfferInput,
    AcceptOfferUseCase,
)
from models import Booking, Institution, TransportRequest, TransportRequestLeg
from models.enums import OfferMode, OfferStatus, RequestStatus
from models.request_offer import RequestOffer
from routes.institution_requests import _apply_return_fields
from services.institutions.mission_schedule import apply_departure_schedule
from services.institutions.transport_request_display import (
    build_transport_request_display_blocks,
)
from services.institutions.transport_request_legs_service import persist_legs
from shared.time_utils import (
    api_scheduled_iso_to_naive_geneva,
    mission_scheduled_to_api_iso,
    now_local,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"EMS P2.5 {uuid.uuid4().hex[:6]}"
    inst.institution_type = "ems"
    db.session.add(inst)
    db.session.flush()
    return inst


def _future_mission_day(*, days_ahead: int = 5) -> date:
    return (now_local() + timedelta(days=days_ahead)).date()


def _naive_dt(mission_day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(mission_day.year, mission_day.month, mission_day.day, hour, minute)


def _naive_iso(mission_day: date, hour: int, minute: int = 0) -> str:
    return _naive_dt(mission_day, hour, minute).strftime("%Y-%m-%dT%H:%M:%S")


def _hhmm_from_iso(iso: str | None) -> str:
    assert iso is not None, "ISO attendu non null"
    return iso[11:16]


def _base_transport_request(
    db,
    institution: Institution,
    *,
    mission_day: date,
    scheduled_time: datetime | None = None,
    multi_stop: bool = False,
    is_round_trip: bool = False,
) -> TransportRequest:
    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.institution = institution
    tr.external_reference = f"P25-{uuid.uuid4().hex[:8]}"
    tr.pickup_location = "Clinique A, 1200 Genève"
    tr.dropoff_location = "Hôpital B, 1205 Genève"
    tr.mission_date = mission_day
    tr.scheduled_time = scheduled_time
    tr.status = RequestStatus.SENT.value
    tr.multi_stop = multi_stop
    tr.is_round_trip = is_round_trip
    tr.billing_intent = "patient"
    db.session.add(tr)
    db.session.flush()
    return tr


def _create_offer(db, transport_request: TransportRequest, company_id: int) -> RequestOffer:
    offer = RequestOffer(
        transport_request_id=transport_request.id,
        company_id=company_id,
        mode=OfferMode.BROADCAST.value,
        status=OfferStatus.PENDING.value,
    )
    db.session.add(offer)
    db.session.flush()
    return offer


# ---------------------------------------------------------------------------
# Cas 1 — Départ institution simple
# ---------------------------------------------------------------------------


class TestCas1InstitutionDepartureSimple:
    """Départ 12:30 → TransportRequest + serialize + projection institution."""

    def test_departure_wall_clock_preserved_end_to_end(
        self, db, requires_postgresql, institution
    ):
        mission_day = _future_mission_day()
        tr = _base_transport_request(db, institution, mission_day=mission_day)

        validated = {
            "mission_date": mission_day.isoformat(),
            "scheduled_time": _naive_iso(mission_day, 12, 30),
            "scheduled_time_type": "departure",
            "pickup_time_confirmed": True,
        }
        apply_departure_schedule(tr, validated)
        db.session.commit()
        db.session.refresh(tr)

        naive_depart = api_scheduled_iso_to_naive_geneva(tr.scheduled_time)
        assert naive_depart == _naive_dt(mission_day, 12, 30)

        serialized = tr.serialize
        assert _hhmm_from_iso(serialized["scheduled_time"]) == "12:30"
        assert serialized["scheduled_time"] == mission_scheduled_to_api_iso(
            tr.scheduled_time
        )

        display = build_transport_request_display_blocks(tr)
        summary = (display.get("scheduling") or {}).get("summary") or ""
        assert "12:30" in summary


# ---------------------------------------------------------------------------
# Cas 2 — Multi-stop
# ---------------------------------------------------------------------------


class TestCas2MultiStopWallClock:
    """Legs A 12:30 / B 13:15 / C 14:00 préservés en API et projection."""

    def test_multi_stop_legs_wall_clock_preserved(
        self, db, requires_postgresql, institution
    ):
        mission_day = _future_mission_day()
        tr = _base_transport_request(
            db,
            institution,
            mission_day=mission_day,
            multi_stop=True,
        )
        tr.route_group_id = str(uuid.uuid4())
        tr.scheduled_time = _naive_dt(mission_day, 8, 0)
        tr.pickup_time_confirmed = True

        legs_data = [
            {
                "sequence_index": 0,
                "route_sequence_number": 1,
                "pickup_location": "Clinique A",
                "dropoff_location": "Hôpital B",
                "scheduled_time": _naive_iso(mission_day, 12, 30),
                "time_confirmed": True,
            },
            {
                "sequence_index": 1,
                "route_sequence_number": 2,
                "pickup_location": "Hôpital B",
                "dropoff_location": "Clinique C",
                "scheduled_time": _naive_iso(mission_day, 13, 15),
                "time_confirmed": True,
            },
            {
                "sequence_index": 2,
                "route_sequence_number": 3,
                "pickup_location": "Clinique C",
                "dropoff_location": "Clinique A",
                "scheduled_time": _naive_iso(mission_day, 14, 0),
                "time_confirmed": True,
            },
        ]
        persist_legs(tr.id, legs_data)
        db.session.commit()

        legs = (
            TransportRequestLeg.query.filter_by(transport_request_id=tr.id)
            .order_by(TransportRequestLeg.sequence_index.asc())
            .all()
        )
        assert len(legs) == 3
        tr.legs = legs

        expected_times = ["12:30", "13:15", "14:00"]
        for leg, expected in zip(legs, expected_times, strict=True):
            leg_payload = leg.serialize()
            assert _hhmm_from_iso(leg_payload["scheduled_time"]) == expected
            hour, minute = (int(expected[:2]), int(expected[3:]))
            assert api_scheduled_iso_to_naive_geneva(leg.scheduled_time) == _naive_dt(
                mission_day, hour, minute
            )

        display = build_transport_request_display_blocks(tr)
        leg_items = display.get("legs") or []
        for item, expected in zip(leg_items, expected_times, strict=True):
            assert _hhmm_from_iso(item.get("scheduled_time")) == expected


# ---------------------------------------------------------------------------
# Cas 3 — Acceptation transporteur + proposed_pickup_time
# ---------------------------------------------------------------------------


class TestCas3AcceptOfferProposedPickupTime:
    """proposed_pickup_time 12:30 → request + booking + booking_summary."""

    def test_proposed_pickup_time_chain_preserved(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        test_company.is_approved = True
        db.session.flush()

        mission_day = _future_mission_day()
        original_depart = _naive_dt(mission_day, 10, 0)
        tr = _base_transport_request(
            db,
            institution,
            mission_day=mission_day,
            scheduled_time=original_depart,
        )
        tr.pickup_time_confirmed = True
        offer = _create_offer(db, tr, test_company.id)
        db.session.commit()

        proposed = _naive_dt(mission_day, 12, 30)
        uc = AcceptOfferUseCase()
        result = uc.execute(
            AcceptOfferInput(
                offer_id=offer.id,
                company_id=test_company.id,
                user_id=test_company.user_id,
                proposed_pickup_time=proposed,
            )
        )
        assert result.success is True, result.error
        db.session.commit()

        db.session.refresh(tr)
        booking = Booking.query.get(result.booking_id)
        assert booking is not None

        assert api_scheduled_iso_to_naive_geneva(tr.scheduled_time) == proposed
        assert api_scheduled_iso_to_naive_geneva(booking.scheduled_time) == proposed

        summary = tr.serialize.get("booking_summary") or {}
        assert _hhmm_from_iso(summary.get("scheduled_time")) == "12:30"

        company_payload = booking.serialize
        assert _hhmm_from_iso(
            mission_scheduled_to_api_iso(booking.scheduled_time)
        ) == "12:30"
        assert company_payload.get("time_formatted", "").startswith("12:30")


# ---------------------------------------------------------------------------
# Cas 4 — Aller-retour
# ---------------------------------------------------------------------------


class TestCas4RoundTripDepartureAndReturn:
    """Départ et retour cohérents après écriture institution."""

    def test_round_trip_departure_and_return_wall_clock(
        self, db, requires_postgresql, institution, test_company, test_client
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        test_company.is_approved = True
        db.session.flush()

        mission_day = _future_mission_day()
        tr = _base_transport_request(
            db,
            institution,
            mission_day=mission_day,
            is_round_trip=True,
        )

        depart_validated = {
            "mission_date": mission_day.isoformat(),
            "scheduled_time": _naive_iso(mission_day, 12, 30),
            "scheduled_time_type": "departure",
            "pickup_time_confirmed": True,
        }
        apply_departure_schedule(tr, depart_validated)

        return_validated = {
            "return_time": _naive_iso(mission_day, 16, 0),
            "return_time_confirmed": True,
        }
        _apply_return_fields(tr, return_validated)
        db.session.commit()
        db.session.refresh(tr)

        serialized = tr.serialize
        assert _hhmm_from_iso(serialized["scheduled_time"]) == "12:30"
        assert _hhmm_from_iso(serialized["return_time"]) == "16:00"
        assert api_scheduled_iso_to_naive_geneva(tr.scheduled_time) == _naive_dt(
            mission_day, 12, 30
        )
        assert api_scheduled_iso_to_naive_geneva(tr.return_time) == _naive_dt(
            mission_day, 16, 0
        )

        offer = _create_offer(db, tr, test_company.id)
        db.session.commit()

        uc = AcceptOfferUseCase()
        result = uc.execute(
            AcceptOfferInput(
                offer_id=offer.id,
                company_id=test_company.id,
                user_id=test_company.user_id,
            )
        )
        assert result.success is True, result.error
        db.session.commit()

        outbound = Booking.query.get(result.booking_id)
        return_booking = Booking.query.get(result.return_booking_id)
        assert outbound is not None
        assert return_booking is not None

        assert _hhmm_from_iso(
            mission_scheduled_to_api_iso(outbound.scheduled_time)
        ) == "12:30"
        assert _hhmm_from_iso(
            mission_scheduled_to_api_iso(return_booking.scheduled_time)
        ) == "16:00"

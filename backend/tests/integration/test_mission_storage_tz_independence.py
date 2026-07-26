# ruff: noqa: I001
"""STOP GATE P3-B — stockage mission indépendant de la TZ process.

Après migration timestamptz → timestamp (heure murale Genève), la persistance
doit rester 12:30 en base et en API quel que soit TZ=UTC ou TZ=Europe/Zurich.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import text

from application.institutions.accept_offer import AcceptOfferInput, AcceptOfferUseCase
from models import Booking, Institution, TransportRequest
from models.enums import OfferMode, OfferStatus, RequestStatus
from models.request_offer import RequestOffer
from services.institutions.mission_schedule import apply_departure_schedule
from services.institutions.transport_request_display import (
    build_transport_request_display_blocks,
)
from shared.time_utils import mission_scheduled_to_api_iso, now_local


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"EMS P3-B {uuid.uuid4().hex[:6]}"
    inst.institution_type = "ems"
    db.session.add(inst)
    db.session.flush()
    return inst


def _future_mission_day(*, days_ahead: int = 5) -> date:
    return (now_local() + timedelta(days=days_ahead)).date()


def _naive_iso(mission_day: date, hour: int, minute: int = 0) -> str:
    return datetime(
        mission_day.year, mission_day.month, mission_day.day, hour, minute
    ).strftime("%Y-%m-%dT%H:%M:%S")


def _hhmm_from_iso(iso: str | None) -> str:
    assert iso is not None, "ISO attendu non null"
    return iso[11:16]


def _read_db_scheduled_time(db, request_id: int) -> datetime:
    row = db.session.execute(
        text("SELECT scheduled_time FROM transport_requests WHERE id = :id"),
        {"id": request_id},
    ).one()
    return row[0]


def _create_offer(
    db, transport_request: TransportRequest, company_id: int
) -> RequestOffer:
    offer = RequestOffer(
        transport_request_id=transport_request.id,
        company_id=company_id,
        mode=OfferMode.BROADCAST.value,
        status=OfferStatus.PENDING.value,
    )
    db.session.add(offer)
    db.session.flush()
    return offer


@pytest.mark.parametrize("tz_env", ["UTC", "Europe/Zurich"])
class TestMissionStorageTzIndependence:
    """Écriture réelle + lecture SQL brute — indépendant de la TZ process."""

    def test_departure_12_30_persisted_and_serialized(
        self, db, requires_postgresql, institution, monkeypatch, tz_env
    ):
        monkeypatch.setenv("TZ", tz_env)

        mission_day = _future_mission_day()
        tr = TransportRequest()
        tr.public_id = str(uuid.uuid4())
        tr.institution_id = institution.id
        tr.institution = institution
        tr.external_reference = f"P3B-{uuid.uuid4().hex[:8]}"
        tr.pickup_location = "Clinique A, 1200 Genève"
        tr.dropoff_location = "Hôpital B, 1205 Genève"
        tr.mission_date = mission_day
        tr.status = RequestStatus.SENT.value
        tr.billing_intent = "patient"
        db.session.add(tr)
        db.session.flush()

        apply_departure_schedule(
            tr,
            {
                "mission_date": mission_day.isoformat(),
                "scheduled_time": _naive_iso(mission_day, 12, 30),
                "scheduled_time_type": "departure",
                "pickup_time_confirmed": True,
            },
        )
        db.session.commit()
        db.session.expire(tr)

        db_value = _read_db_scheduled_time(db, tr.id)
        assert db_value == datetime(
            mission_day.year, mission_day.month, mission_day.day, 12, 30
        )
        assert db_value.tzinfo is None

        db.session.refresh(tr)
        serialized = tr.serialize
        assert _hhmm_from_iso(serialized["scheduled_time"]) == "12:30"
        assert serialized["scheduled_time"] == mission_scheduled_to_api_iso(
            tr.scheduled_time
        )

        display = build_transport_request_display_blocks(tr)
        summary = (display.get("scheduling") or {}).get("summary") or ""
        assert "12:30" in summary

    def test_accept_offer_enterprise_projection_12_30(
        self,
        db,
        requires_postgresql,
        institution,
        test_company,
        test_client,
        monkeypatch,
        tz_env,
    ):
        if not test_company or not test_client:
            pytest.skip("test_company and test_client required")

        monkeypatch.setenv("TZ", tz_env)
        test_company.is_approved = True
        db.session.flush()

        mission_day = _future_mission_day()
        tr = TransportRequest()
        tr.public_id = str(uuid.uuid4())
        tr.institution_id = institution.id
        tr.institution = institution
        tr.external_reference = f"P3B-OFFER-{uuid.uuid4().hex[:8]}"
        tr.pickup_location = "Clinique A, 1200 Genève"
        tr.dropoff_location = "Hôpital B, 1205 Genève"
        tr.mission_date = mission_day
        tr.status = RequestStatus.SENT.value
        tr.billing_intent = "patient"
        db.session.add(tr)
        db.session.flush()

        apply_departure_schedule(
            tr,
            {
                "mission_date": mission_day.isoformat(),
                "scheduled_time": _naive_iso(mission_day, 12, 30),
                "scheduled_time_type": "departure",
                "pickup_time_confirmed": True,
            },
        )
        offer = _create_offer(db, tr, test_company.id)
        db.session.commit()

        db_value = _read_db_scheduled_time(db, tr.id)
        assert db_value.hour == 12
        assert db_value.minute == 30

        proposed = datetime(
            mission_day.year, mission_day.month, mission_day.day, 12, 30
        )
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

        booking = Booking.query.get(result.booking_id)
        assert booking is not None

        db.session.refresh(tr)
        assert (
            _hhmm_from_iso(mission_scheduled_to_api_iso(tr.scheduled_time)) == "12:30"
        )
        assert (
            _hhmm_from_iso(mission_scheduled_to_api_iso(booking.scheduled_time))
            == "12:30"
        )

        summary = tr.serialize.get("booking_summary") or {}
        assert _hhmm_from_iso(summary.get("scheduled_time")) == "12:30"

        company_payload = booking.serialize
        assert company_payload.get("time_formatted", "").startswith("12:30")

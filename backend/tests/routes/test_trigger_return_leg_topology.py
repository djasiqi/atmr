# ruff: noqa: I001
"""Garde-fou trigger-return : résolution topologie institution (return leg)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from application.companies.reservations.resolve_return_target import (
    ReturnTopologyError,
    resolve_existing_return_target,
)
from models import Booking, Institution, TransportRequest, TransportRequestLeg
from models.enums import BookingStatus, RequestStatus
from repositories.booking_repository import BookingRepository
from tests.routes.test_companies import _auth_headers

pytest_plugins = ["tests.routes.test_companies"]


def _future_return_time(hours: int = 8) -> str:
    return (datetime.now(UTC) + timedelta(days=3, hours=hours)).isoformat()


def _build_leg_topology(
    db,
    *,
    company,
    client_user,
    client,
    return_scheduled: datetime | None = None,
    return_time_confirmed: bool = False,
    return_billing: dict | None = None,
):
    """Repro #4464-like : aller + return leg booking (is_return=false)."""
    route_group_id = str(uuid.uuid4())
    outbound = Booking()
    outbound.user_id = client_user.id
    outbound.client_id = client.id
    outbound.company_id = company.id
    outbound.customer_name = "Patient Leg"
    outbound.pickup_location = "Clinique A"
    outbound.dropoff_location = "HUG"
    outbound.scheduled_time = datetime.now(UTC) + timedelta(days=2)
    outbound.time_confirmed = True
    outbound.status = BookingStatus.ACCEPTED
    outbound.amount = Decimal("80.00")
    outbound.is_return = False
    outbound.route_group_id = route_group_id
    outbound.route_sequence_number = 1
    db.session.add(outbound)
    db.session.flush()

    return_booking = Booking()
    return_booking.user_id = client_user.id
    return_booking.client_id = client.id
    return_booking.company_id = company.id
    return_booking.customer_name = "Patient Leg"
    return_booking.pickup_location = "HUG"
    return_booking.dropoff_location = "Clinique A"
    return_booking.scheduled_time = return_scheduled
    return_booking.time_confirmed = return_time_confirmed
    return_booking.status = BookingStatus.PENDING
    return_booking.amount = Decimal("80.00")
    return_booking.is_return = False
    return_booking.route_group_id = route_group_id
    return_booking.route_sequence_number = 2
    if return_billing:
        return_booking.billing_party_id = return_billing.get("billing_party_id")
        return_booking.billed_to_type = return_billing.get("billed_to_type")
        return_booking.billed_to_company_id = return_billing.get("billed_to_company_id")
    db.session.add(return_booking)
    db.session.flush()

    institution = Institution()
    institution.public_id = str(uuid.uuid4())
    institution.name = f"Inst Leg {uuid.uuid4().hex[:6]}"
    institution.institution_type = "clinic"
    institution.address = "Rue Inst 1"
    institution.billing_address = "Rue Inst 1"
    db.session.add(institution)
    db.session.flush()

    tr = TransportRequest()
    tr.public_id = str(uuid.uuid4())
    tr.institution_id = institution.id
    tr.created_by_user_id = client_user.id
    tr.created_by_display_name = "Test Institution User"
    tr.external_reference = f"TR-LEG-{uuid.uuid4().hex[:8]}"
    tr.pickup_location = outbound.pickup_location
    tr.dropoff_location = return_booking.dropoff_location
    tr.mission_date = outbound.scheduled_time.date()
    tr.scheduled_time = outbound.scheduled_time
    tr.pickup_time_confirmed = True
    tr.status = RequestStatus.CONVERTED.value
    tr.multi_stop = True
    tr.return_to_institution = True
    tr.route_group_id = route_group_id
    tr.booking_id = outbound.id
    db.session.add(tr)
    db.session.flush()

    outbound_leg = TransportRequestLeg(
        transport_request_id=tr.id,
        sequence_index=0,
        route_sequence_number=1,
        pickup_location=outbound.pickup_location,
        dropoff_location=outbound.dropoff_location,
        is_return_stop=False,
        booking_id=outbound.id,
    )
    return_leg = TransportRequestLeg(
        transport_request_id=tr.id,
        sequence_index=1,
        route_sequence_number=2,
        pickup_location=return_booking.pickup_location,
        dropoff_location=return_booking.dropoff_location,
        is_return_stop=True,
        booking_id=return_booking.id,
        scheduled_time=return_scheduled,
        time_confirmed=return_time_confirmed,
    )
    db.session.add_all([outbound_leg, return_leg])
    db.session.commit()

    return {
        "transport_request": tr,
        "outbound": outbound,
        "return_booking": return_booking,
        "outbound_leg": outbound_leg,
        "return_leg": return_leg,
    }


@pytest.fixture
def company_headers(app, companies_world):
    world = companies_world
    return _auth_headers(
        app, world["company_user"], role="company", company_id=world["company"].id
    )


class TestResolveReturnTarget:
    def test_outbound_resolves_to_return_leg_booking(self, db, companies_world):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        resolution = resolve_existing_return_target(
            topo["outbound"],
            company_id=world["company"].id,
            booking_repo=BookingRepository(),
        )
        assert resolution.action == "modify_leg_return"
        assert resolution.source == "institution_return_leg"
        assert resolution.target_booking is not None
        assert resolution.target_booking.id == topo["return_booking"].id

    def test_return_leg_booking_resolves_modify_current(self, db, companies_world):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        resolution = resolve_existing_return_target(
            topo["return_booking"],
            company_id=world["company"].id,
        )
        assert resolution.action == "modify_current"
        assert resolution.source == "institution_return_leg"

    def test_classic_child_return(self, db, companies_world):
        world = companies_world
        outbound = world["booking"]
        child = Booking()
        child.user_id = world["client_user"].id
        child.client_id = world["client"].id
        child.company_id = world["company"].id
        child.customer_name = outbound.customer_name
        child.pickup_location = outbound.dropoff_location
        child.dropoff_location = outbound.pickup_location
        child.scheduled_time = datetime.now(UTC) + timedelta(days=4)
        child.status = BookingStatus.ACCEPTED
        child.amount = outbound.amount
        child.is_return = True
        child.parent_booking_id = outbound.id
        db.session.add(child)
        db.session.commit()

        resolution = resolve_existing_return_target(
            outbound,
            company_id=world["company"].id,
        )
        assert resolution.action == "modify_existing_classic_return"
        assert resolution.source == "classic_child_return"
        assert resolution.target_booking.id == child.id

    def test_no_topology_create_new(self, db, companies_world):
        world = companies_world
        resolution = resolve_existing_return_target(
            world["booking"],
            company_id=world["company"].id,
        )
        assert resolution.action == "create_new"
        assert resolution.source == "none"

    def test_return_leg_without_booking_raises(self, db, companies_world):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        topo["return_leg"].booking_id = None
        db.session.commit()

        with pytest.raises(ReturnTopologyError) as exc:
            resolve_existing_return_target(
                topo["outbound"],
                company_id=world["company"].id,
            )
        assert exc.value.http_status == 409


class TestTriggerReturnLegTopology:
    def test_trigger_on_outbound_modifies_return_leg_booking_no_duplicate(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        outbound_id = topo["outbound"].id
        return_id = topo["return_booking"].id
        before_count = Booking.query.count()
        return_time = _future_return_time(14)

        resp = client.post(
            f"/api/v1/companies/me/reservations/{outbound_id}/trigger-return",
            headers=company_headers,
            json={"return_time": return_time, "time_confirmed": True},
        )
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        assert body["return_booking"]["id"] == return_id
        assert Booking.query.count() == before_count

        db.session.refresh(topo["return_booking"])
        assert topo["return_booking"].scheduled_time is not None
        assert topo["return_booking"].time_confirmed is True
        assert topo["return_booking"].status == BookingStatus.ACCEPTED
        assert topo["return_booking"].is_return is False
        assert (
            Booking.query.filter_by(
                parent_booking_id=outbound_id, is_return=True
            ).count()
            == 0
        )

    def test_return_leg_without_schedule(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        assert topo["return_booking"].scheduled_time is None

        resp = client.post(
            f"/api/v1/companies/me/reservations/{topo['outbound'].id}/trigger-return",
            headers=company_headers,
            json={"return_time": _future_return_time(16), "time_confirmed": True},
        )
        assert resp.status_code == 200, resp.get_json()
        db.session.refresh(topo["return_booking"])
        assert topo["return_booking"].scheduled_time is not None

    def test_return_leg_reschedule_idempotent(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        first_time = datetime.now(UTC) + timedelta(days=4, hours=10)
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
            return_scheduled=first_time,
            return_time_confirmed=True,
        )
        return_id = topo["return_booking"].id
        second_time = _future_return_time(18)

        resp = client.post(
            f"/api/v1/companies/me/reservations/{topo['outbound'].id}/trigger-return",
            headers=company_headers,
            json={"return_time": second_time, "time_confirmed": True},
        )
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json()["return_booking"]["id"] == return_id
        db.session.refresh(topo["return_booking"])
        assert topo["return_booking"].id == return_id

    def test_billing_fields_immutable_on_leg_return(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        billing = {
            "billing_party_id": None,
            "billed_to_type": "clinic",
            "billed_to_company_id": world["company"].id,
        }
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
            return_billing=billing,
        )

        resp = client.post(
            f"/api/v1/companies/me/reservations/{topo['outbound'].id}/trigger-return",
            headers=company_headers,
            json={"return_time": _future_return_time(12), "time_confirmed": True},
        )
        assert resp.status_code == 200, resp.get_json()
        db.session.refresh(topo["return_booking"])
        assert topo["return_booking"].billing_party_id is None
        assert topo["return_booking"].billed_to_type == "clinic"
        assert topo["return_booking"].billed_to_company_id == world["company"].id

    def test_classic_round_trip_existing_child(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        outbound = world["booking"]
        child = Booking()
        child.user_id = world["client_user"].id
        child.client_id = world["client"].id
        child.company_id = world["company"].id
        child.customer_name = outbound.customer_name
        child.pickup_location = outbound.dropoff_location
        child.dropoff_location = outbound.pickup_location
        child.scheduled_time = datetime.now(UTC) + timedelta(days=5)
        child.status = BookingStatus.ACCEPTED
        child.amount = outbound.amount
        child.is_return = True
        child.parent_booking_id = outbound.id
        db.session.add(child)
        db.session.commit()
        child_id = child.id
        before_count = Booking.query.count()

        resp = client.post(
            f"/api/v1/companies/me/reservations/{outbound.id}/trigger-return",
            headers=company_headers,
            json={"return_time": _future_return_time(11), "time_confirmed": True},
        )
        assert resp.status_code == 200, resp.get_json()
        assert resp.get_json()["return_booking"]["id"] == child_id
        assert Booking.query.count() == before_count

    def test_classic_round_trip_creates_when_missing(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        outbound = world["booking"]
        before_count = Booking.query.count()

        resp = client.post(
            f"/api/v1/companies/me/reservations/{outbound.id}/trigger-return",
            headers=company_headers,
            json={"return_time": _future_return_time(9), "time_confirmed": True},
        )
        assert resp.status_code == 200, resp.get_json()
        assert Booking.query.count() == before_count + 1
        created = Booking.query.filter_by(
            parent_booking_id=outbound.id, is_return=True
        ).one()
        assert resp.get_json()["return_booking"]["id"] == created.id

    def test_inconsistent_return_leg_no_fallback_create(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        topo["return_leg"].booking_id = None
        db.session.commit()
        before_count = Booking.query.count()

        resp = client.post(
            f"/api/v1/companies/me/reservations/{topo['outbound'].id}/trigger-return",
            headers=company_headers,
            json={"return_time": _future_return_time(13), "time_confirmed": True},
        )
        assert resp.status_code == 409, resp.get_json()
        assert Booking.query.count() == before_count
        assert (
            Booking.query.filter_by(
                parent_booking_id=topo["outbound"].id, is_return=True
            ).count()
            == 0
        )

    def test_repeated_calls_single_logical_return(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        outbound_id = topo["outbound"].id
        return_id = topo["return_booking"].id
        before_count = Booking.query.count()
        payload = {"return_time": _future_return_time(15), "time_confirmed": True}

        for _ in range(2):
            resp = client.post(
                f"/api/v1/companies/me/reservations/{outbound_id}/trigger-return",
                headers=company_headers,
                json=payload,
            )
            assert resp.status_code == 200, resp.get_json()
            assert resp.get_json()["return_booking"]["id"] == return_id

        assert Booking.query.count() == before_count
        assert (
            Booking.query.filter_by(
                parent_booking_id=outbound_id, is_return=True
            ).count()
            == 0
        )

    def test_api_returns_actual_return_leg_booking(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )

        resp = client.post(
            f"/api/v1/companies/me/reservations/{topo['outbound'].id}/trigger-return",
            headers=company_headers,
            json={"urgent": True, "minutes_offset": 30},
        )
        assert resp.status_code == 200, resp.get_json()
        returned = resp.get_json()["return_booking"]
        assert returned["id"] == topo["return_booking"].id
        assert returned["pickup_location"] == topo["return_booking"].pickup_location

    def test_listing_exposes_trip_flags_return_leg_topology(
        self, client, db, companies_world, company_headers
    ):
        world = companies_world
        topo = _build_leg_topology(
            db,
            company=world["company"],
            client_user=world["client_user"],
            client=world["client"],
        )
        day = topo["outbound"].scheduled_time.date().isoformat()

        resp = client.get(
            f"/api/v1/companies/me/reservations?date={day}&page=1&per_page=50",
            headers=company_headers,
        )
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        reservations = body.get("reservations") or body.get("data", {}).get(
            "reservations", []
        )
        by_id = {r["id"]: r for r in reservations}

        outbound_payload = by_id.get(topo["outbound"].id)
        return_payload = by_id.get(topo["return_booking"].id)
        assert outbound_payload is not None
        assert return_payload is not None
        assert outbound_payload["trip_flags"]["return_leg"] is False
        assert return_payload["trip_flags"]["return_leg"] is True
        assert return_payload["is_return"] is False

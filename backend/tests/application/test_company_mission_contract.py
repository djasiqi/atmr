"""Contrat canonique de réservation entreprise."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pytest
from marshmallow import ValidationError

from application.companies.reservations.company_mission import (
    CompanyMissionError,
    apply_company_mission_mobile_read,
    canonical_payload_hash,
    company_mission_mobile_read,
    company_mission_read_payload,
    hash_canonical_request,
    normalize_route_steps,
    preview_company_mission_pricing,
    resolve_company_mission_pricing,
    split_equal_cents,
)
from application.companies.reservations.create_manual_booking import (
    CreateManualBookingError,
    CreateManualBookingUseCase,
)
from application.companies.reservations.update_reservation import (
    UpdateCompanyReservationUseCase,
)
from models.company_booking_mission import (
    CompanyBookingRouteStep,
    CompanyManualBookingRequest,
    CompanyManualBookingRequestOccurrence,
)
from schemas.company_schemas import ManualBookingCreateSchema
from services.adapters.mobile_booking_adapter import (
    map_mobile_ride_payload_to_manual_booking_payload,
)


def _next_slot(hour: int = 9) -> datetime:
    moment = datetime.now() + timedelta(days=2)
    return moment.replace(hour=hour, minute=0, second=0, microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _steps(start: datetime, *, destinations: int = 1, with_return: bool = False) -> list[dict]:
    steps: list[dict] = [
        {
            "position": 0,
            "kind": "pickup",
            "location": "Rue du Départ 1",
            "lat": 46.2,
            "lon": 6.14,
            "arrival_at": None,
            "departure_at": _iso(start),
        }
    ]
    for index in range(destinations):
        arrival = start + timedelta(hours=index + 1)
        last = index == destinations - 1
        departure = None
        if not last or with_return:
            departure = _iso(arrival + timedelta(minutes=30))
        steps.append(
            {
                "position": index + 1,
                "kind": "destination",
                "location": f"Destination {index + 1}",
                "lat": 46.21 + index / 100,
                "lon": 6.15,
                "arrival_at": _iso(arrival),
                "departure_at": departure,
                "destination_kind": "other",
            }
        )
    if with_return:
        last_departure = datetime.fromisoformat(steps[-1]["departure_at"])
        steps.append(
            {
                "position": len(steps),
                "kind": "return",
                "location": "Adresse différente du départ",
                "lat": 1.0,
                "lon": 2.0,
                "arrival_at": _iso(last_departure + timedelta(hours=1)),
                "departure_at": None,
            }
        )
    return steps


def _base_payload(**overrides):
    start = _next_slot()
    payload = {
        "client_id": 1,
        "route_steps": _steps(start),
        "pricing_mode": "manual",
        "idempotency_key": "cle-test",
        "segment_amounts": [{"from_position": 0, "to_position": 1, "amount": "45.00"}],
        "mission_type": "patient_transport",
    }
    payload.update(overrides)
    return payload


def test_illegal_routes_are_rejected():
    start = _next_slot()
    pickup = {
        "position": 0,
        "kind": "pickup",
        "location": "A",
        "departure_at": _iso(start),
    }
    destination = {
        "position": 1,
        "kind": "destination",
        "location": "B",
        "arrival_at": _iso(start + timedelta(hours=1)),
        "destination_kind": "other",
    }
    retour = {
        "position": 1,
        "kind": "return",
        "location": "A",
        "arrival_at": _iso(start + timedelta(hours=2)),
    }
    with pytest.raises(CompanyMissionError):
        normalize_route_steps([pickup, retour], mission_type="patient_transport")
    late_return = dict(retour, position=2)
    with pytest.raises(CompanyMissionError):
        normalize_route_steps(
            [pickup, late_return, dict(destination, position=1)],
            mission_type="patient_transport",
        )
    with pytest.raises(CompanyMissionError):
        normalize_route_steps(
            [pickup, dict(destination, position=2)],
            mission_type="patient_transport",
        )


def test_night_crossing_keeps_both_dates():
    steps = normalize_route_steps(
        [
            {
                "position": 0,
                "kind": "pickup",
                "location": "A",
                "departure_at": "2026-10-02T23:30:00",
            },
            {
                "position": 1,
                "kind": "destination",
                "location": "B",
                "arrival_at": "2026-10-03T00:15:00",
                "destination_kind": "medical",
            },
        ],
        mission_type="patient_transport",
    )
    assert steps[0]["departure_at"] == datetime(2026, 10, 2, 23, 30)
    assert steps[1]["arrival_at"] == datetime(2026, 10, 3, 0, 15)


def test_chronology_and_delivery_rules():
    start = _next_slot()
    steps = _steps(start)
    steps[1]["departure_at"] = _iso(start)
    steps[1]["arrival_at"] = _iso(start + timedelta(hours=2))
    with pytest.raises(CompanyMissionError):
        normalize_route_steps(steps, mission_type="patient_transport")

    missing_kind = _steps(start)
    missing_kind[1]["destination_kind"] = None
    with pytest.raises(CompanyMissionError):
        normalize_route_steps(missing_kind, mission_type="patient_transport")

    delivery = _steps(start)
    delivery[1]["destination_kind"] = "medical"
    normalized = normalize_route_steps(delivery, mission_type="material_delivery")
    assert normalized[1]["destination_kind"] is None


def test_undefined_return_time_is_accepted():
    start = _next_slot(14)
    steps = _steps(start, with_return=True)
    steps[1]["departure_at"] = None
    steps[1]["establishment"] = "HUG"
    steps[1]["service"] = "Radiologie"
    steps[1]["doctor"] = "Docteur Dupont"
    steps[1]["destination_kind"] = "medical"
    steps[-1]["arrival_at"] = None
    normalized = normalize_route_steps(steps, mission_type="patient_transport")
    assert normalized[1]["departure_at"] is None
    assert normalized[-1]["arrival_at"] is None
    assert normalized[-1]["location"] == "Rue du Départ 1"


def test_return_address_is_replaced_and_hashes_match():
    start = _next_slot()
    first = _steps(start, with_return=True)
    second = _steps(start, with_return=True)
    second[-1]["location"] = "Encore une autre adresse"
    second[-1]["lat"] = 9
    left = normalize_route_steps(first, mission_type="patient_transport")
    right = normalize_route_steps(second, mission_type="patient_transport")
    assert left[-1]["location"] == "Rue du Départ 1"
    assert right[-1]["location"] == left[-1]["location"]
    assert right[-1]["latitude"] == left[-1]["latitude"]
    payload_a = _base_payload(route_steps=first, idempotency_key="a")
    payload_b = _base_payload(route_steps=second, idempotency_key="b")
    payload_a["segment_amounts"] = [
        {"from_position": 0, "to_position": 1, "amount": "45"},
        {"from_position": 1, "to_position": 2, "amount": "45.00"},
    ]
    payload_b["segment_amounts"] = [
        {"from_position": 0, "to_position": 1, "amount": "45.00"},
        {"from_position": 1, "to_position": 2, "amount": "45"},
    ]
    assert hash_canonical_request(payload_a) == hash_canonical_request(payload_b)
    assert canonical_payload_hash({"idempotency_key": "x", "a": 1}) == canonical_payload_hash(
        {"a": 1}
    )


def test_schema_discriminator():
    schema = ManualBookingCreateSchema()
    with pytest.raises(ValidationError):
        schema.load({"client_id": 1, "route_steps": _steps(_next_slot())})
    with pytest.raises(ValidationError):
        schema.load(
            {
                "client_id": 1,
                "route_steps": _steps(_next_slot()),
                "pricing_mode": "manual",
                "idempotency_key": "cle",
                "pickup_location": "A",
                "segment_amounts": [
                    {"from_position": 0, "to_position": 1, "amount": "45.00"}
                ],
            }
        )
    with pytest.raises(ValidationError):
        schema.load(
            {
                "client_id": 1,
                "route_steps": _steps(_next_slot(), with_return=True),
                "pricing_mode": "manual",
                "idempotency_key": "cle",
                "is_round_trip": True,
            }
        )
    legacy = schema.load(
        {
            "client_id": 1,
            "pickup_location": "A",
            "dropoff_location": "B",
            "scheduled_time": _iso(_next_slot()),
            "amount": 45,
        }
    )
    assert "idempotency_key" not in legacy or not legacy.get("idempotency_key")
    assert "route_steps" not in legacy or not legacy.get("route_steps")


def test_pricing_modes_and_preview_series(monkeypatch):
    amounts = iter([Decimal("90.00"), Decimal("90.00"), Decimal("100.00")])

    def _fake(**_kwargs):
        return next(amounts), False

    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        _fake,
    )
    start = _next_slot()
    client = type("Client", (), {"preferential_rate": None})()
    preview = preview_company_mission_pricing(
        company_id=1,
        client=client,
        validated_data={
            "client_id": 1,
            "route_steps": _steps(start),
            "pricing_mode": "automatic",
            "is_recurring": True,
            "recurrence_type": "daily",
            "occurrences": 3,
            "mission_type": "patient_transport",
        },
    )
    assert [row["total"] for row in preview["occurrences"]] == ["90.00", "90.00", "100.00"]
    assert preview["series_total"] == "280.00"
    with pytest.raises(CompanyMissionError):
        preview_company_mission_pricing(
            company_id=1,
            client=client,
            validated_data={
                "client_id": 1,
                "route_steps": _steps(start),
                "pricing_mode": "automatic",
                "segment_amounts": [
                    {"from_position": 0, "to_position": 1, "amount": "10"}
                ],
                "mission_type": "patient_transport",
            },
        )

    priced = resolve_company_mission_pricing(
        company_id=1,
        client=client,
        steps=normalize_route_steps(_steps(start, destinations=3), mission_type="patient_transport"),
        pricing_mode="manual",
        segment_amounts=[
            {"from_position": 0, "to_position": 1, "amount": "45"},
            {"from_position": 1, "to_position": 2, "amount": "60"},
            {"from_position": 2, "to_position": 3, "amount": "45.00"},
        ],
        preferential_amount=None,
    )
    assert [row["amount"] for row in priced] == ["45.00", "60.00", "45.00"]

    parts = split_equal_cents(Decimal("100.00"), 3)
    assert parts == [Decimal("33.33"), Decimal("33.33"), Decimal("33.34")]
    assert sum(parts) == Decimal("100.00")
    assert split_equal_cents(Decimal("120.00"), 3) == [
        Decimal("40.00"),
        Decimal("40.00"),
        Decimal("40.00"),
    ]


def test_simple_round_trip_keeps_existing_split(monkeypatch):
    def _fake(**kwargs):
        assert kwargs["is_round_trip"] is True
        return Decimal("70.00"), True

    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        _fake,
    )
    client = type("Client", (), {"preferential_rate": None})()
    steps = normalize_route_steps(
        _steps(_next_slot(), with_return=True),
        mission_type="patient_transport",
    )
    priced = resolve_company_mission_pricing(
        company_id=1,
        client=client,
        steps=steps,
        pricing_mode="automatic",
        segment_amounts=None,
        preferential_amount=None,
    )
    assert [row["amount"] for row in priced] == ["35.00", "35.00"]


def test_mobile_mapper_stays_on_legacy_contract():
    mapped = map_mobile_ride_payload_to_manual_booking_payload(
        {
            "pickup_address": "Rue A",
            "dropoff_address": "Rue B",
            "scheduled_time": _iso(_next_slot()),
            "is_return": True,
        }
    )
    assert mapped["pickup_location"] == "Rue A"
    assert "route_steps" not in mapped
    assert "pricing_mode" not in mapped
    assert "idempotency_key" not in mapped


def _company_and_client(db):
    from models import Client, Company, User
    from models.enums import UserRole

    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"co_{suffix}"
    owner.email = f"co-{suffix}@test.ch"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123", force_change=False)
    db.session.add(owner)
    db.session.flush()

    company = Company()
    company.name = f"Co {suffix}"
    company.address = "Rue 1"
    company.contact_email = f"c-{suffix}@test.ch"
    company.user_id = owner.id
    db.session.add(company)
    db.session.flush()

    person = User()
    person.username = f"cli_{suffix}"
    person.email = f"cli-{suffix}@test.ch"
    person.role = UserRole.client
    person.first_name = "Ada"
    person.last_name = "Lovelace"
    person.public_id = str(uuid.uuid4())
    person.set_password("password123", force_change=False)
    db.session.add(person)
    db.session.flush()

    client = Client()
    client.user_id = person.id
    client.company_id = company.id
    client.default_billed_to_type = "patient"
    db.session.add(client)
    db.session.flush()
    return company, client, person


def _execute(company, client, user, payload):
    with (
        patch("services.platform_billing.capabilities.assert_billing_capability_allowed"),
        patch(
            "services.billing.client_stay_resolver.find_active_stay_for_client",
            return_value=None,
        ),
        patch(
            "application.companies.reservations.create_manual_booking._geocode_with_nominatim",
            return_value=(46.2, 6.1),
        ),
        patch("services.geolocation.osrm._route") as mock_route,
    ):
        mock_route.return_value = {
            "code": "Ok",
            "routes": [{"duration": 600, "distance": 4000}],
        }
        return CreateManualBookingUseCase().execute(
            company_id=company.id,
            validated_data=payload,
            client=client,
            user=user,
        )


def test_canonical_create_read_update_and_idempotency(db, monkeypatch):
    from models import Booking

    company, client, user = _company_and_client(db)
    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        lambda **_kwargs: (Decimal("45.00"), False),
    )
    start = _next_slot(11)
    steps = _steps(start, destinations=3)
    payload = {
        "client_id": client.id,
        "route_steps": steps,
        "pricing_mode": "automatic",
        "idempotency_key": f"auto-{uuid.uuid4().hex}",
        "mission_type": "patient_transport",
        "passenger_name": "Passager Test",
    }
    preview = preview_company_mission_pricing(
        company_id=company.id, client=client, validated_data=payload
    )
    result = _execute(company, client, user, payload)
    assert len(result.created_outbounds) == 3
    assert sum(float(row.amount) for row in result.created_outbounds) == 135
    assert all(float(row.amount) > 0 for row in result.created_outbounds)
    assert [row["amount"] for row in preview["segments"]] == [
        f"{float(row.amount):.2f}" for row in result.created_outbounds
    ]
    anchor = result.created_outbounds[0]
    assert anchor.passenger_name == "Passager Test"
    assert anchor.customer_name == "Ada Lovelace"
    assert anchor.pricing_mode == "automatic"
    assert result.created_outbounds[1].pricing_mode is None
    persisted = CompanyBookingRouteStep.query.filter_by(anchor_booking_id=anchor.id).count()
    assert persisted == 4

    secondary = result.created_outbounds[1]
    read_secondary = company_mission_read_payload(secondary)
    assert read_secondary["mission_anchor_booking_id"] == anchor.id
    assert read_secondary["route_steps_source"] == "canonical"
    assert len(read_secondary["route_steps"]) == 4
    assert secondary.serialize["route_steps_source"] == "canonical"
    assert secondary.serialize["mission_anchor_booking_id"] == anchor.id

    for booking in result.created_outbounds:
        updated = UpdateCompanyReservationUseCase().execute(
            booking, validated_data={"scheduled_time": _iso(_next_slot(15))}
        )
        assert updated.ok is False
        assert updated.status_code == 400
    assert CompanyBookingRouteStep.query.filter_by(anchor_booking_id=anchor.id).count() == 4

    replay = _execute(company, client, user, payload)
    assert [row.id for row in replay.created_outbounds] == [
        row.id for row in result.created_outbounds
    ]
    assert (
        CompanyManualBookingRequest.query.filter_by(company_id=company.id).count() == 1
    )

    conflict = dict(payload)
    conflict["passenger_name"] = "Autre"
    with pytest.raises(CreateManualBookingError) as exc:
        _execute(company, client, user, conflict)
    assert exc.value.status_code == 409
    assert exc.value.error_code == "IDEMPOTENCY_CONFLICT"
    assert Booking.query.get(anchor.id).passenger_name == "Passager Test"


def test_manual_preferential_return_and_legacy(db):
    from models import Booking

    company, client, user = _company_and_client(db)
    start = _next_slot(14)
    manual_steps = _steps(start, destinations=2, with_return=True)
    manual = {
        "client_id": client.id,
        "route_steps": manual_steps,
        "pricing_mode": "manual",
        "idempotency_key": f"man-{uuid.uuid4().hex}",
        "segment_amounts": [
            {"from_position": 0, "to_position": 1, "amount": "45.00"},
            {"from_position": 1, "to_position": 2, "amount": "60.00"},
            {"from_position": 2, "to_position": 3, "amount": "45.00"},
        ],
        "passenger_name": "Manuel",
    }
    created = _execute(company, client, user, manual)
    amounts = sorted(float(row.amount) for row in created.created_outbounds + created.created_returns)
    assert amounts == [45.0, 45.0, 60.0]
    retour = created.created_returns[0]
    assert retour.is_return is True
    assert retour.parent_booking_id == created.created_outbounds[0].id
    assert retour.dropoff_location == "Rue du Départ 1"
    assert created.created_outbounds[0].is_round_trip is True
    blocked = UpdateCompanyReservationUseCase().execute(
        retour, validated_data={"scheduled_time": _iso(_next_slot(16))}
    )
    assert blocked.status_code == 400

    preferential_steps = _steps(_next_slot(16), destinations=3)
    preferential = {
        "client_id": client.id,
        "route_steps": preferential_steps,
        "pricing_mode": "preferential",
        "idempotency_key": f"pref-{uuid.uuid4().hex}",
        "preferential_amount": "100.00",
    }
    pref = _execute(company, client, user, preferential)
    pref_amounts = [Decimal(str(row.amount)) for row in pref.created_outbounds]
    assert pref_amounts == [Decimal("33.33"), Decimal("33.33"), Decimal("33.34")]

    legacy = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "Aller A",
            "dropoff_location": "Aller B",
            "scheduled_time": _iso(_next_slot(18)),
            "is_round_trip": True,
            "return_date": (_next_slot(18) + timedelta(hours=3)).date().isoformat(),
            "return_time": _iso(_next_slot(18) + timedelta(hours=3)),
            "amount": 45,
        },
    )
    assert len(legacy.created_outbounds) == 1
    assert len(legacy.created_returns) == 1
    assert (
        CompanyBookingRouteStep.query.filter_by(
            anchor_booking_id=legacy.created_outbounds[0].id
        ).count()
        == 0
    )
    synthesized = company_mission_read_payload(legacy.created_outbounds[0])
    assert synthesized["route_steps_source"] == "legacy_synthesized"
    assert [step["kind"] for step in synthesized["route_steps"]] == [
        "pickup",
        "destination",
        "return",
    ]
    assert (
        CompanyManualBookingRequest.query.filter_by(
            company_id=company.id, idempotency_key=None
        ).count()
        == 0
    )
    legacy_edit = UpdateCompanyReservationUseCase().execute(
        legacy.created_outbounds[0],
        validated_data={"pickup_location": "Aller A modifié"},
    )
    assert legacy_edit.ok is True
    assert Booking.query.get(legacy.created_outbounds[0].id).pickup_location == "Aller A modifié"


def test_recurrence_is_one_transaction(db, monkeypatch):
    company, client, user = _company_and_client(db)
    start = _next_slot(8)
    while start.weekday() != 0:
        start += timedelta(days=1)
    calls = {"n": 0}
    real = __import__(
        "application.companies.reservations.company_mission",
        fromlist=["_materialize_occurrence"],
    )._materialize_occurrence

    def _boom(**kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise CompanyMissionError("échec sur la troisième occurrence")
        return real(**kwargs)

    monkeypatch.setattr(
        "application.companies.reservations.company_mission._materialize_occurrence",
        _boom,
    )
    payload = {
        "client_id": client.id,
        "route_steps": _steps(start),
        "pricing_mode": "manual",
        "idempotency_key": f"rec-{uuid.uuid4().hex}",
        "segment_amounts": [{"from_position": 0, "to_position": 1, "amount": "45.00"}],
        "is_recurring": True,
        "recurrence_type": "custom",
        "recurrence_days": [0],
        "occurrences": 4,
    }
    with pytest.raises(CreateManualBookingError):
        _execute(company, client, user, payload)
    assert (
        CompanyManualBookingRequest.query.filter_by(company_id=company.id).count() == 0
    )

    monkeypatch.setattr(
        "application.companies.reservations.company_mission._materialize_occurrence",
        real,
    )
    company, client, user = _company_and_client(db)
    payload = dict(payload, client_id=client.id, idempotency_key=f"rec-{uuid.uuid4().hex}")
    created = _execute(company, client, user, payload)
    assert len(created.created_outbounds) == 4
    assert (
        CompanyManualBookingRequest.query.filter_by(
            company_id=company.id, idempotency_key=payload["idempotency_key"]
        ).count()
        == 1
    )
    request_row = CompanyManualBookingRequest.query.filter_by(
        idempotency_key=payload["idempotency_key"]
    ).one()
    assert (
        CompanyManualBookingRequestOccurrence.query.filter_by(request_id=request_row.id).count()
        == 4
    )
    replay = _execute(company, client, user, payload)
    assert len(replay.created_outbounds) == 4


def _assert_put_blocked(booking):
    updated = UpdateCompanyReservationUseCase().execute(
        booking, validated_data={"scheduled_time": _iso(_next_slot(19))}
    )
    assert updated.ok is False
    assert updated.status_code == 400


def test_undefined_return_keeps_time_open_and_swaps_place_details(db, monkeypatch):
    company, client, user = _company_and_client(db)
    client.domicile_address = "Rue du Départ 1"
    client.floor = "3"
    client.door_code = "A12"
    db.session.flush()
    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        lambda **_kwargs: (Decimal("45.00"), False),
    )
    start = _next_slot(14)
    steps = _steps(start, with_return=True)
    steps[0]["access_notes"] = "Sonnette"
    steps[1]["departure_at"] = None
    steps[1]["establishment"] = "HUG"
    steps[1]["service"] = "Radiologie"
    steps[1]["doctor"] = "Docteur Dupont"
    steps[1]["destination_kind"] = "medical"
    steps[1]["access_notes"] = "Entrée principale"
    steps[-1]["arrival_at"] = None
    steps[-1]["access_notes"] = "Sonnette"
    created = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": steps,
            "pricing_mode": "manual",
            "idempotency_key": f"open-return-{uuid.uuid4().hex}",
            "segment_amounts": [
                {"from_position": 0, "to_position": 1, "amount": "45.00"},
                {"from_position": 1, "to_position": 2, "amount": "45.00"},
            ],
        },
    )
    outbound = created.created_outbounds[0]
    retour = created.created_returns[0]
    assert outbound.time_confirmed is True
    assert outbound.scheduled_time is not None
    assert outbound.medical_facility == "HUG"
    assert outbound.pickup_floor == "3"
    assert outbound.pickup_door_code == "A12"
    assert retour.time_confirmed is False
    assert retour.scheduled_time is None
    assert retour.medical_facility == "HUG"
    assert retour.hospital_service == "Radiologie"
    assert retour.doctor_name == "Docteur Dupont"
    assert retour.pickup_access_notes == "Entrée principale"
    assert retour.dropoff_access_notes == "Sonnette"
    assert retour.dropoff_floor == "3"
    assert retour.dropoff_door_code == "A12"
    assert retour.pickup_floor is None


def test_detach_anchor_lets_round_trip_be_deleted(db, monkeypatch):
    from sqlalchemy import text

    company, client, user = _company_and_client(db)
    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        lambda **_kwargs: (Decimal("45.00"), False),
    )
    created = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": _steps(_next_slot(14), with_return=True),
            "pricing_mode": "manual",
            "idempotency_key": f"del-{uuid.uuid4().hex}",
            "segment_amounts": [
                {"from_position": 0, "to_position": 1, "amount": "45.00"},
                {"from_position": 1, "to_position": 2, "amount": "45.00"},
            ],
        },
    )
    anchor = created.created_outbounds[0]
    retour = created.created_returns[0]
    anchor_id = int(anchor.id)
    return_id = int(retour.id)
    company_id = int(company.id)
    assert (
        CompanyManualBookingRequestOccurrence.query.filter_by(
            anchor_booking_id=anchor_id
        ).count()
        == 1
    )
    from application.companies.reservations.company_mission import (
        detach_company_mission_anchor,
    )

    detach_company_mission_anchor(anchor_id)
    db.session.execute(text("DELETE FROM booking WHERE id = :id"), {"id": return_id})
    db.session.execute(text("DELETE FROM booking WHERE id = :id"), {"id": anchor_id})
    db.session.flush()
    left = db.session.execute(
        text("SELECT COUNT(*) FROM booking WHERE id IN (:a, :b)"),
        {"a": anchor_id, "b": return_id},
    ).scalar()
    assert int(left or 0) == 0
    occurrences = db.session.execute(
        text(
            "SELECT COUNT(*) FROM company_manual_booking_request_occurrences "
            "WHERE anchor_booking_id = :id"
        ),
        {"id": anchor_id},
    ).scalar()
    assert int(occurrences or 0) == 0
    requests_left = db.session.execute(
        text(
            "SELECT COUNT(*) FROM company_manual_booking_requests WHERE company_id = :id"
        ),
        {"id": company_id},
    ).scalar()
    assert int(requests_left or 0) == 0


def test_route_shapes_legacy_simple_and_secondary_reads(db, monkeypatch):
    """A→B, A→B→A, A→B→C, A→B→C→A, lecture secondaire, legacy simple."""
    from models import Booking

    company, client, user = _company_and_client(db)
    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        lambda **_kwargs: (Decimal("45.00"), False),
    )
    start = _next_slot(12)

    direct = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": _steps(start),
            "pricing_mode": "automatic",
            "idempotency_key": f"ab-{uuid.uuid4().hex}",
            "passenger_name": "Passager AB",
            "requester_name": "Contact AB",
            "requester_phone": "0220000000",
            "requester_service": "Accueil",
        },
    )
    assert len(direct.created_outbounds) == 1
    assert direct.created_returns == []
    anchor_ab = direct.created_outbounds[0]
    assert anchor_ab.route_group_id is None
    assert CompanyBookingRouteStep.query.filter_by(anchor_booking_id=anchor_ab.id).count() == 2
    assert anchor_ab.requester_name == "Contact AB"
    _assert_put_blocked(anchor_ab)

    round_trip = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": _steps(_next_slot(13), with_return=True),
            "pricing_mode": "manual",
            "idempotency_key": f"aba-{uuid.uuid4().hex}",
            "segment_amounts": [
                {"from_position": 0, "to_position": 1, "amount": "40.00"},
                {"from_position": 1, "to_position": 2, "amount": "40.00"},
            ],
        },
    )
    assert len(round_trip.created_outbounds) == 1
    assert len(round_trip.created_returns) == 1
    assert round_trip.created_outbounds[0].route_group_id is None
    assert round_trip.created_returns[0].parent_booking_id == round_trip.created_outbounds[0].id
    return_read = company_mission_read_payload(round_trip.created_returns[0])
    assert return_read["route_steps_source"] == "canonical"
    assert return_read["mission_anchor_booking_id"] == round_trip.created_outbounds[0].id
    assert [step["kind"] for step in return_read["route_steps"]] == [
        "pickup",
        "destination",
        "return",
    ]
    _assert_put_blocked(round_trip.created_returns[0])

    multi = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": _steps(_next_slot(15), destinations=2),
            "pricing_mode": "manual",
            "idempotency_key": f"abc-{uuid.uuid4().hex}",
            "segment_amounts": [
                {"from_position": 0, "to_position": 1, "amount": "45.00"},
                {"from_position": 1, "to_position": 2, "amount": "50.00"},
            ],
        },
    )
    assert len(multi.created_outbounds) == 2
    assert multi.created_returns == []
    assert multi.created_outbounds[0].route_group_id
    assert multi.created_outbounds[1].route_group_id == multi.created_outbounds[0].route_group_id
    secondary = company_mission_read_payload(multi.created_outbounds[1])
    assert secondary["mission_anchor_booking_id"] == multi.created_outbounds[0].id
    assert len(secondary["route_steps"]) == 3
    _assert_put_blocked(multi.created_outbounds[0])
    _assert_put_blocked(multi.created_outbounds[1])

    loop = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": _steps(_next_slot(17), destinations=2, with_return=True),
            "pricing_mode": "manual",
            "idempotency_key": f"abca-{uuid.uuid4().hex}",
            "segment_amounts": [
                {"from_position": 0, "to_position": 1, "amount": "45.00"},
                {"from_position": 1, "to_position": 2, "amount": "60.00"},
                {"from_position": 2, "to_position": 3, "amount": "45.00"},
            ],
        },
    )
    assert len(loop.created_outbounds) == 2
    assert len(loop.created_returns) == 1
    assert loop.created_returns[0].parent_booking_id == loop.created_outbounds[0].id
    loop_return = company_mission_read_payload(loop.created_returns[0])
    assert loop_return["mission_anchor_booking_id"] == loop.created_outbounds[0].id
    assert [step["kind"] for step in loop_return["route_steps"]] == [
        "pickup",
        "destination",
        "destination",
        "return",
    ]
    _assert_put_blocked(loop.created_returns[0])

    legacy = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "Legacy A",
            "dropoff_location": "Legacy B",
            "scheduled_time": _iso(_next_slot(18)),
            "amount": 45,
        },
    )
    assert len(legacy.created_outbounds) == 1
    assert legacy.created_returns == []
    assert (
        CompanyBookingRouteStep.query.filter_by(
            anchor_booking_id=legacy.created_outbounds[0].id
        ).count()
        == 0
    )
    edited = UpdateCompanyReservationUseCase().execute(
        legacy.created_outbounds[0],
        validated_data={"dropoff_location": "Legacy B modifié"},
    )
    assert edited.ok is True
    assert Booking.query.get(legacy.created_outbounds[0].id).dropoff_location == "Legacy B modifié"


def _read_mission(company, client, user, monkeypatch, steps):
    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        lambda **_kwargs: (Decimal("40.00"), False),
    )
    return _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "route_steps": steps,
            "pricing_mode": "automatic",
            "idempotency_key": f"read-{uuid.uuid4().hex}",
            "mission_type": "patient_transport",
            "notes_medical": "Note d'ancre",
            "requester_name": "Camille",
            "requester_phone": "0220000000",
            "needs_assistance": True,
            "wheelchair_client_has": True,
        },
    )


def test_mobile_read_covers_one_way_round_trip_and_group(db, monkeypatch):
    from models import Booking, BookingStatus

    company, client, user = _company_and_client(db)
    one_way = _read_mission(company, client, user, monkeypatch, _steps(_next_slot(8)))
    anchor_one = one_way.created_outbounds[0]
    read_one = company_mission_mobile_read(anchor_one)
    assert read_one["route_steps_source"] == "canonical"
    assert read_one["mission_segment_index"] == 1
    assert read_one["mission_segment_count"] == 1
    assert read_one["notes_medical"] == "Note d'ancre"
    assert read_one["requester_name"] == "Camille"
    assert read_one["wheelchair_client_has"] is True

    simple = _read_mission(
        company,
        client,
        user,
        monkeypatch,
        _steps(_next_slot(10), with_return=True),
    )
    outbound = simple.created_outbounds[0]
    retour = simple.created_returns[0]
    assert outbound.route_group_id is None
    assert retour.parent_booking_id == outbound.id
    read_out = company_mission_mobile_read(outbound)
    read_back = company_mission_mobile_read(retour)
    assert read_out["mission_anchor_booking_id"] == outbound.id
    assert read_back["mission_anchor_booking_id"] == outbound.id
    assert read_out["route_steps"] == read_back["route_steps"]
    assert [step["kind"] for step in read_out["route_steps"]] == [
        "pickup",
        "destination",
        "return",
    ]
    assert (read_out["mission_segment_index"], read_out["mission_segment_count"]) == (1, 2)
    assert (read_back["mission_segment_index"], read_back["mission_segment_count"]) == (2, 2)

    grouped = _read_mission(
        company,
        client,
        user,
        monkeypatch,
        _steps(_next_slot(12), destinations=2, with_return=True),
    )
    first, second = grouped.created_outbounds
    third = grouped.created_returns[0]
    assert first.route_group_id
    before_steps = CompanyBookingRouteStep.query.filter_by(anchor_booking_id=first.id).count()
    second.notes_medical = "Note du segment, pas de la mission"
    db.session.flush()
    reads = [
        company_mission_mobile_read(first),
        company_mission_mobile_read(second),
        company_mission_mobile_read(third),
    ]
    assert [item["mission_segment_index"] for item in reads] == [1, 2, 3]
    assert {item["mission_segment_count"] for item in reads} == {3}
    assert {item["mission_anchor_booking_id"] for item in reads} == {first.id}
    assert reads[0]["route_steps"] == reads[1]["route_steps"] == reads[2]["route_steps"]
    assert [step["kind"] for step in reads[1]["route_steps"]] == [
        "pickup",
        "destination",
        "destination",
        "return",
    ]
    assert reads[1]["notes_medical"] == "Note d'ancre"
    assert reads[1]["requester_name"] == "Camille"
    assert CompanyBookingRouteStep.query.filter_by(anchor_booking_id=first.id).count() == before_steps
    assert not any(isinstance(obj, CompanyBookingRouteStep) for obj in db.session.new)

    second.status = BookingStatus.CANCELED
    db.session.flush()
    cancelled_read = company_mission_mobile_read(third)
    assert cancelled_read["mission_segment_index"] == 3
    assert cancelled_read["mission_segment_count"] == 3

    other_company, other_client, other_user = _company_and_client(db)
    intruder = Booking()
    intruder.company_id = other_company.id
    intruder.client_id = other_client.id
    intruder.user_id = other_user.id
    intruder.route_group_id = first.route_group_id
    intruder.route_sequence_number = 50
    intruder.customer_name = "Intrus"
    intruder.pickup_location = "Hors tenant"
    intruder.dropoff_location = "Hors tenant"
    intruder.scheduled_time = first.scheduled_time
    intruder.status = BookingStatus.ACCEPTED
    intruder.amount = 1
    db.session.add(intruder)
    db.session.flush()
    assert company_mission_mobile_read(second)["mission_segment_count"] == 3
    assert company_mission_mobile_read(intruder)["mission_segment_count"] == 1

    summary = {
        "id": str(second.id),
        "route": {
            "pickup_address": second.pickup_location,
            "dropoff_address": second.dropoff_location,
        },
        "status": "accepted",
    }
    apply_company_mission_mobile_read(summary, second)
    assert summary["id"] == str(second.id)
    assert summary["route"]["pickup_address"] == second.pickup_location
    assert summary["route"]["dropoff_address"] == second.dropoff_location
    assert summary["status"] == "accepted"
    assert summary["mission_segment_index"] == 2
    assert summary["mission_anchor_booking_id"] == first.id


def test_mobile_ride_detail_endpoint_puts_mission_on_summary(app, db, monkeypatch):
    """GET /company_mobile/dispatch/v1/rides/:id — champs sur summary, pas l'enveloppe."""
    company, client, user = _company_and_client(db)
    grouped = _read_mission(
        company,
        client,
        user,
        monkeypatch,
        _steps(_next_slot(13), destinations=2, with_return=True),
    )
    first = grouped.created_outbounds[0]
    second = grouped.created_outbounds[1]
    monkeypatch.setattr(
        "routes.company_mobile_dispatch._get_company_context",
        lambda: (company, int(company.id)),
    )
    from routes.company_mobile_dispatch import MobileRideDetail

    view = MobileRideDetail()
    handler = view.get
    while hasattr(handler, "__wrapped__"):
        handler = handler.__wrapped__
    body, status = handler(view, str(second.id))
    assert status == 200
    assert "mission_anchor_booking_id" not in body
    assert "route_steps" not in body
    summary = body["summary"]
    assert summary["id"] == str(second.id)
    assert summary["route"]["pickup_address"] == second.pickup_location
    assert summary["route"]["dropoff_address"] == second.dropoff_location
    assert summary["mission_anchor_booking_id"] == first.id
    assert summary["route_steps_source"] == "canonical"
    assert summary["mission_segment_index"] == 2
    assert summary["mission_segment_count"] == 3
    assert [step["kind"] for step in summary["route_steps"]] == [
        "pickup",
        "destination",
        "destination",
        "return",
    ]
    assert summary["notes_medical"] == "Note d'ancre"


def test_mobile_legacy_read_synthesizes_without_writing(db, monkeypatch):
    company, client, user = _company_and_client(db)
    monkeypatch.setattr(
        "application.companies.reservations.company_mission._price_one_way_segment",
        lambda **_kwargs: (Decimal("40.00"), False),
    )
    simple = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "Legacy A",
            "dropoff_location": "Legacy B",
            "scheduled_time": _iso(_next_slot(14)),
            "amount": 45,
        },
    )
    booking = simple.created_outbounds[0]
    before = CompanyBookingRouteStep.query.filter_by(anchor_booking_id=booking.id).count()
    read_simple = company_mission_mobile_read(booking)
    assert read_simple["route_steps_source"] == "legacy_synthesized"
    assert [step["kind"] for step in read_simple["route_steps"]] == ["pickup", "destination"]
    assert (read_simple["mission_segment_index"], read_simple["mission_segment_count"]) == (1, 1)
    assert CompanyBookingRouteStep.query.filter_by(anchor_booking_id=booking.id).count() == before

    round_trip = _execute(
        company,
        client,
        user,
        {
            "client_id": client.id,
            "pickup_location": "Legacy A",
            "dropoff_location": "Legacy B",
            "scheduled_time": _iso(_next_slot(15)),
            "is_round_trip": True,
            "return_time": _iso(_next_slot(15) + timedelta(hours=3)),
            "amount": 90,
        },
    )
    outbound = round_trip.created_outbounds[0]
    assert round_trip.created_returns
    retour = round_trip.created_returns[0]
    read_out = company_mission_mobile_read(outbound)
    read_back = company_mission_mobile_read(retour)
    assert read_out["route_steps_source"] == "legacy_synthesized"
    assert read_back["route_steps_source"] == "legacy_synthesized"
    assert [step["kind"] for step in read_out["route_steps"]] == [
        "pickup",
        "destination",
        "return",
    ]
    assert read_out["route_steps"] == read_back["route_steps"]
    assert (read_out["mission_segment_index"], read_out["mission_segment_count"]) == (1, 2)
    assert (read_back["mission_segment_index"], read_back["mission_segment_count"]) == (2, 2)

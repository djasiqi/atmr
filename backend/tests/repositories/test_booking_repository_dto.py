from __future__ import annotations

from types import SimpleNamespace

from repositories.booking_repository import BookingRepository


def test_booking_repository_to_dto_converts_model_fields() -> None:
    repo = BookingRepository()

    fake_model = SimpleNamespace(
        id=1,
        company_id=2,
        executing_company_id=2,
        client_id=3,
        user_id=4,
        driver_id=5,
        customer_name="A",
        pickup_location="X",
        dropoff_location="Y",
        scheduled_time=None,
        boarded_at=None,
        completed_at=None,
        status="PENDING",
        amount=10.0,
        pickup_lat=46.2,
        pickup_lon=6.1,
        dropoff_lat=46.3,
        dropoff_lon=6.2,
        distance_meters=1000,
        duration_seconds=60,
        is_round_trip=False,
        is_return=False,
        is_urgent=False,
        time_confirmed=True,
        parent_booking_id=None,
    )

    dto = repo._to_dto(fake_model)  # type: ignore[arg-type]
    assert dto.id == 1
    assert dto.company_id == 2
    assert dto.pickup_lat == 46.2
    assert dto.dropoff_lon == 6.2

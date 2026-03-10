from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from marshmallow import ValidationError

from application.bookings.create_booking import CreateBookingUseCase
from domain.bookings.commands import CreateBookingCommand


class _FakeClientRepo:
    def __init__(self, company_id: int = 7) -> None:
        self._company_id = company_id

    def find_by_id(self, _client_id: int):  # type: ignore[no-untyped-def]
        return SimpleNamespace(id=1, company_id=self._company_id)


class _FakeCompanyRepo:
    def find_model_by_id(self, _company_id: int):  # type: ignore[no-untyped-def]
        return None


class _FakeBookingWriter:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def create_and_commit(self, **kwargs):  # type: ignore[no-untyped-def]
        # Simule un booking persisté (sans DB)
        self.last_kwargs = kwargs
        return SimpleNamespace(id=123, company_id=7)


class _FakeGeocoding:
    def geocode_address(self, _address: str, *, country: str | None = None):  # type: ignore[no-untyped-def]
        _ = country
        return {"lat": 46.2, "lon": 6.1}


def test_create_booking_use_case_publishes_booking_created_event(monkeypatch) -> None:
    published: list[dict[str, Any]] = []

    import application.bookings.create_booking as mod

    def fake_publish_event(evt):  # type: ignore[no-untyped-def]
        published.append(evt.to_dict())

    monkeypatch.setattr(mod, "publish_event", fake_publish_event)
    monkeypatch.setattr(
        mod,
        "resolve_pickup_admin",
        lambda **_kwargs: {
            "token": "commune:6630",
            "canton_code": "GE",
            "source": "db",
            "confidence": "authoritative",
            "label": "Anieres (GE)",
        },
    )
    import services.billing.client_stay_resolver as stay_mod

    monkeypatch.setattr(stay_mod, "find_active_stay_for_client", lambda **_kwargs: None)
    monkeypatch.setattr(stay_mod, "get_clinic_address_for_stay", lambda _stay: None)

    writer = _FakeBookingWriter()
    uc = CreateBookingUseCase(
        client_repo=_FakeClientRepo(company_id=7),  # type: ignore[arg-type]
        company_lookup=_FakeCompanyRepo(),  # type: ignore[arg-type]
        booking_writer=writer,  # type: ignore[arg-type]
        geocoding_service=_FakeGeocoding(),  # type: ignore[arg-type]
        distance_duration_fn=lambda _p, _d: (60, 1000),
        fallback_coords_fn=lambda _company: (46.2044, 6.1432),
    )

    cmd = CreateBookingCommand(
        user_id=1,
        client_id=1,
        data={
            "customer_name": "A",
            "pickup_location": "X",
            "dropoff_location": "Y",
            "scheduled_time": datetime.now(UTC).isoformat(timespec="seconds"),
            "amount": 10.0,
            "is_round_trip": False,
        },
    )

    booking = uc.execute(cmd)
    assert getattr(booking, "id", None) == 123

    assert published
    assert published[0]["event_type"] == "BookingCreatedEvent"
    assert published[0]["booking_id"] == 123
    assert published[0]["company_id"] == 7
    assert writer.last_kwargs["pickup_admin_source"] in {"db", "geoadmin", "photon", "unknown"}
    assert writer.last_kwargs["dropoff_admin_source"] in {"db", "geoadmin", "photon", "unknown"}
    assert "pickup_admin_resolved_at" in writer.last_kwargs
    assert "dropoff_admin_resolved_at" in writer.last_kwargs


def test_create_booking_use_case_invalid_scheduled_time_raises_value_error() -> None:
    uc = CreateBookingUseCase(
        client_repo=_FakeClientRepo(),  # type: ignore[arg-type]
        company_lookup=_FakeCompanyRepo(),  # type: ignore[arg-type]
        booking_writer=_FakeBookingWriter(),  # type: ignore[arg-type]
        geocoding_service=_FakeGeocoding(),  # type: ignore[arg-type]
        distance_duration_fn=lambda _p, _d: (60, 1000),
        fallback_coords_fn=lambda _company: (46.2044, 6.1432),
    )

    with pytest.raises(ValidationError, match="scheduled_time"):
        uc.execute(
            CreateBookingCommand(
                user_id=1,
                client_id=1,
                data={
                    "customer_name": "A",
                    "pickup_location": "X",
                    "dropoff_location": "Y",
                    "scheduled_time": "not-a-date",
                    "amount": 10.0,
                },
            )
        )

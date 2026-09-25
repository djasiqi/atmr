"""7B.2 — CreateBooking ignore totalement maximum_accepted_amount client."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from flask import Flask

from domain.bookings.commands import CreateBookingCommand
from models.enums import ClientType
from services.legal.portal_double_validation import FLOW_DOUBLE_VALIDATION_V2
from services.pricing.portal_carrier_ceiling import PortalCarrierCeiling
from tests.helpers.create_booking_use_case import CreateBookingUseCase


class _PortalClientRepo:
    def find_by_id(self, _client_id: int):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            id=1,
            company_id=None,
            client_type=ClientType.PORTAL,
            preferential_rate=None,
        )


class _Writer:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    def create_and_commit(self, **kwargs):  # type: ignore[no-untyped-def]
        self.last_kwargs = kwargs
        return SimpleNamespace(
            id=701,
            company_id=kwargs.get("company_id"),
            status="pending",
            amount=kwargs["amount"],
            price_amount=kwargs.get("price_amount"),
            customer_name=kwargs.get("customer_name"),
            pickup_location=kwargs.get("pickup_location"),
            dropoff_location=kwargs.get("dropoff_location"),
            scheduled_time=kwargs.get("scheduled_time"),
            is_round_trip=kwargs.get("is_round_trip"),
            wheelchair_need=kwargs.get("wheelchair_need"),
            billed_to_type=kwargs.get("billed_to_type"),
            portal_contract_flow=None,
        )


class _Geocoding:
    def geocode_address(self, address: str, *, country: str | None = None):  # type: ignore[no-untyped-def]
        _ = address, country
        return {"lat": 46.2, "lon": 6.1}


def test_create_booking_ignores_client_maximum_9999(
    monkeypatch: pytest.MonkeyPatch,
    app: Flask,
) -> None:
    """Cas 5 : payload maximum_accepted_amount=9999 → plafond serveur 52."""
    import application.bookings.create_booking as mod
    import services.billing.client_stay_resolver as stay_mod

    monkeypatch.setattr(mod, "publish_event", lambda _evt: None)
    monkeypatch.setattr(
        mod,
        "resolve_pickup_admin",
        lambda **_k: {
            "token": "commune:6630",
            "canton_code": "GE",
            "source": "db",
            "confidence": "authoritative",
            "label": "Geneve",
        },
    )
    monkeypatch.setattr(mod, "geo_unit_id_from_pickup_admin_token", lambda _t: None)
    monkeypatch.setattr(stay_mod, "find_active_stay_for_client", lambda **_k: None)
    monkeypatch.setattr(stay_mod, "get_clinic_address_for_stay", lambda _s: None)
    monkeypatch.setattr(
        "services.legal.portal_terms_status.assert_portal_terms_current",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "services.auth.portal_phone_verification.assert_portal_phone_verified",
        lambda **_k: None,
    )
    monkeypatch.setattr(
        "services.legal.portal_double_validation.resolve_portal_contract_flow",
        lambda **_k: FLOW_DOUBLE_VALIDATION_V2,
    )

    server_ceiling = PortalCarrierCeiling(
        maximum_accepted_amount=52.0,
        currency="CHF",
        pricing_calculated_at="2026-09-24T15:00:00+00:00",
        eligible_carrier_count=3,
        quoted_carrier_count=3,
        quotes=[],
        excluded=[],
        distance_meters=12000,
    )
    monkeypatch.setattr(
        "services.pricing.portal_carrier_ceiling.compute_portal_carrier_ceiling",
        lambda **_k: server_ceiling,
    )

    recorded: dict[str, Any] = {}

    def _fake_record(**kwargs: Any):
        recorded.update(kwargs)
        return SimpleNamespace(id=1)

    monkeypatch.setattr(
        "services.legal.record_booking_contract_event.record_portal_booking_created_event",
        _fake_record,
    )

    writer = _Writer()
    uc = CreateBookingUseCase(
        client_repo=_PortalClientRepo(),  # type: ignore[arg-type]
        company_lookup=SimpleNamespace(find_model_by_id=lambda _cid: None),  # type: ignore[arg-type]
        booking_writer=writer,  # type: ignore[arg-type]
        geocoding_service=_Geocoding(),  # type: ignore[arg-type]
        distance_duration_fn=lambda _p, _d: (1200, 15000),
        company_creation_gate_fn=lambda _cid: None,
        billing_capability_gate_fn=lambda _cid: None,
        fallback_coords_fn=lambda _company: (46.2, 6.1),
    )

    with app.app_context():
        # Transaction déjà gérée par les fixtures de test → nullcontext côté use case
        # si in_transaction() est vrai ; sinon begin() ouvre un contexte.
        uc.execute(
            CreateBookingCommand(
                user_id=42,
                client_id=1,
                data={
                    "customer_name": "Client Prive",
                    "pickup_location": "Rue du Port 1, 1200 Geneve",
                    "dropoff_location": "Rue de la Gare 2, 1200 Geneve",
                    "scheduled_time": datetime.now(UTC).isoformat(timespec="seconds"),
                    "amount": 40.0,
                    "is_round_trip": False,
                    # Tentative côté client — doit être ignorée
                    "maximum_accepted_amount": 9999,
                },
            )
        )

    assert recorded.get("maximum_accepted_amount") == 52.0
    evidence = recorded.get("pricing_ceiling_evidence") or {}
    assert evidence["pricing_ceiling"]["maximum_accepted_amount"] == 52.0
    assert recorded.get("maximum_accepted_amount") != 9999

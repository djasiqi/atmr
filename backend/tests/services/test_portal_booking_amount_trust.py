"""Montant persisté d'une réservation PORTAL, et absence d'encaissement Saferpay.

Le body ``amount`` reste écrit tel quel : aucune grille serveur ne fait
autorité, et l'estimation n'est pas promue en prix contractuel. Ce montant
ne doit plus ouvrir un paiement Saferpay pour un compte PORTAL.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from domain.bookings.commands import CreateBookingCommand
from models.enums import BookingStatus, ClientType
from services.auth.portal_phone_verification import (
    assert_portal_phone_verified as real_assert_portal_can_confirm,
)
from services.booking.portal_platform_payment import (
    PortalPlatformPaymentForbidden,
)
from services.saferpay.payment_page import create_saferpay_payment_page_initialize
from tests.helpers.create_booking_use_case import CreateBookingUseCase

BACKEND_ROOT = Path(__file__).resolve().parents[2]

# Montant « preview » de référence. Le use case de création ne le reçoit pas
# et ne le recalcule pas : il ne sert qu'à montrer l'écart avec le body.
SERVER_PREVIEW_CHF = 90.0
BODY_AMOUNTS = (0.50, 1.00, SERVER_PREVIEW_CHF - 10, SERVER_PREVIEW_CHF + 10)


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
            id=501,
            company_id=kwargs.get("company_id"),
            status="pending",
            amount=kwargs["amount"],
            price_amount=kwargs.get("price_amount"),
        )


class _Geocoding:
    def geocode_address(self, address: str, *, country: str | None = None):  # type: ignore[no-untyped-def]
        _ = address, country
        return {"lat": 46.2, "lon": 6.1}


def _allow_verified_portal(**kwargs: Any) -> None:
    """Garde-fou réel, avec un utilisateur dont le téléphone est déjà vérifié."""
    verified = SimpleNamespace(
        id=kwargs["user_id"], phone_verified_at=datetime.now(UTC)
    )
    real_assert_portal_can_confirm(
        user_id=kwargs["user_id"],
        client=kwargs["client"],
        user_loader=lambda _uid: verified,
    )


def _patch_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
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
        _allow_verified_portal,
    )


def _execute(body_amount: float) -> _Writer:
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
    uc.execute(
        CreateBookingCommand(
            user_id=42,
            client_id=1,
            data={
                "customer_name": "Client Prive",
                "pickup_location": "Rue du Port 1, 1200 Geneve",
                "dropoff_location": "Rue de la Gare 2, 1200 Geneve",
                "scheduled_time": datetime.now(UTC).isoformat(timespec="seconds"),
                "amount": body_amount,
                "is_round_trip": False,
            },
        )
    )
    return writer


@pytest.mark.parametrize("body_amount", BODY_AMOUNTS)
def test_portal_body_amount_is_stored_but_does_not_open_saferpay(
    monkeypatch: pytest.MonkeyPatch, body_amount: float
) -> None:
    _patch_side_effects(monkeypatch)
    writer = _execute(body_amount)
    persisted = float(writer.last_kwargs["amount"])

    http_calls = {"n": 0}
    monkeypatch.setattr(
        "services.saferpay.payment_page.saferpay_post_json",
        lambda *_a, **_k: http_calls.__setitem__("n", http_calls["n"] + 1),
    )
    portal_client = SimpleNamespace(id=1, client_type=ClientType.PORTAL)
    booking = SimpleNamespace(
        id=501,
        client_id=1,
        client=portal_client,
        status=BookingStatus.PENDING,
        amount=persisted,
        company_id=None,
        is_return=False,
        return_trip=None,
    )

    with pytest.raises(PortalPlatformPaymentForbidden):
        create_saferpay_payment_page_initialize(
            booking=booking,
            user=SimpleNamespace(id=42),
            client=portal_client,
        )

    assert writer.last_kwargs["price_amount"] is None
    assert writer.last_kwargs["company_id"] is None
    assert persisted == body_amount
    assert persisted != SERVER_PREVIEW_CHF or body_amount == SERVER_PREVIEW_CHF
    assert http_calls["n"] == 0


def test_portal_amount_trust_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le navigateur peut encore écrire booking.amount ; il ne peut pas encaisser."""
    _patch_side_effects(monkeypatch)
    stored_matches_body: list[bool] = []
    for body_amount in BODY_AMOUNTS:
        writer = _execute(body_amount)
        stored_matches_body.append(float(writer.last_kwargs["amount"]) == body_amount)

    assert all(stored_matches_body) is True


def _post_calls_shared_helper(source: str, class_name: str) -> bool:
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if not isinstance(item, ast.FunctionDef) or item.name != "post":
                continue
            return any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "execute_client_booking_creation"
                for n in ast.walk(item)
            )
    return False


def test_portal_create_routes_share_one_trust_boundary() -> None:
    clients = (BACKEND_ROOT / "routes" / "clients.py").read_text(encoding="utf-8")
    bookings = (BACKEND_ROOT / "routes" / "bookings.py").read_text(encoding="utf-8")
    assert _post_calls_shared_helper(clients, "ClientBookings")
    assert _post_calls_shared_helper(clients, "ClientMyBookings")
    assert _post_calls_shared_helper(bookings, "CreateBooking")

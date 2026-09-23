"""Verrou : le client privé PORTAL n'est pas encaissé via Saferpay.

Les parcours invité et client TRANSPORT restent hors de ce refus.
AUTH-SMS-02 (téléphone non vérifié) n'est pas modifié.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from models.enums import BookingStatus, ClientType
from services.auth.portal_phone_verification import (
    PortalPhoneVerificationRequired,
    assert_portal_phone_verified,
)
from services.booking.portal_platform_payment import (
    should_hold_client_booking_for_platform_payment,
)
from services.saferpay.payment_page import create_saferpay_payment_page_initialize

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_portal_patient_billing_does_not_hold_for_platform_payment() -> None:
    portal = SimpleNamespace(client_type=ClientType.PORTAL)
    transport = SimpleNamespace(client_type=ClientType.TRANSPORT)

    assert should_hold_client_booking_for_platform_payment(portal, "patient") is False
    assert should_hold_client_booking_for_platform_payment(portal, None) is False
    assert should_hold_client_booking_for_platform_payment(transport, "patient") is True
    assert should_hold_client_booking_for_platform_payment(transport, "insurance") is False


def test_transport_checkout_still_enters_saferpay_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.saferpay.payment_page.saferpay_configured",
        lambda: False,
    )
    client = SimpleNamespace(id=2, client_type=ClientType.TRANSPORT)
    booking = SimpleNamespace(
        id=3,
        client_id=2,
        client=client,
        status=BookingStatus.PENDING,
        amount=40,
        company_id=7,
    )

    with pytest.raises(RuntimeError, match="n'est pas configuré"):
        create_saferpay_payment_page_initialize(
            booking=booking,
            user=SimpleNamespace(id=8),
            client=client,
        )


def test_unverified_portal_phone_still_blocks_confirmation() -> None:
    with pytest.raises(PortalPhoneVerificationRequired) as exc:
        assert_portal_phone_verified(
            user_id=1,
            client=SimpleNamespace(client_type=ClientType.PORTAL),
            user_loader=lambda _uid: SimpleNamespace(phone_verified_at=None),
        )
    assert exc.value.code == "phone_verification_required"


def test_guest_saferpay_path_is_unchanged() -> None:
    guest = (BACKEND_ROOT / "services" / "guest_saferpay.py").read_text(encoding="utf-8")
    auth = (BACKEND_ROOT / "routes" / "auth.py").read_text(encoding="utf-8")
    assert "portal_platform_payment" not in guest
    assert "initialize_guest_saferpay" in guest
    assert "/public/guest-booking/saferpay/initialize" in auth
    assert "/public/guest-booking/saferpay/assert" in auth

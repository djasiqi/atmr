"""Téléphone PORTAL vérifié une fois : plus d'OTP par réservation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from flask_jwt_extended import create_access_token

from models import ActivationSession, Client, User
from models.enums import ClientType, UserRole
from routes import auth


def _headers(app, user: User) -> dict[str, str]:
    claims = {
        "role": UserRole.CLIENT.value,
        "company_id": None,
        "driver_id": None,
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _make_portal_user(db, *, pending: bool = False) -> tuple[User, Client]:
    unique = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"portal_{unique}"
    user.email = f"portal_{unique}@example.com"
    user.role = UserRole.CLIENT
    user.public_id = str(uuid.uuid4())
    user.phone = "+41768190077"
    user.account_status = "pending_activation" if pending else "active"
    user.phone_verified_at = None
    user.set_password("Password123!", force_change=False)
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.is_active = not pending
    client.contact_email = user.email
    db.session.add(client)
    db.session.flush()
    return user, client


def _valid_booking_payload() -> dict[str, object]:
    return {
        "customer_name": "Philippe Test",
        "pickup_location": "Rue de la Gare 1, 1000 Lausanne",
        "dropoff_location": "Avenue de la Plage 10, 1000 Lausanne",
        "scheduled_time": (datetime.now(UTC) + timedelta(days=1))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "amount": 50.0,
    }


def test_channel_requirements_email_only_when_email_present(app, monkeypatch):
    user = SimpleNamespace(email="a@test.ch", phone="+41768190077")
    monkeypatch.setattr(
        auth,
        "User",
        SimpleNamespace(query=SimpleNamespace(get=lambda _uid: user)),
    )
    session = SimpleNamespace(user_id=1)
    with app.app_context():
        assert auth._activation_channel_requirements(session) == (True, False)


def test_login_after_email_without_phone_verified(client, db, app):
    user, portal_client = _make_portal_user(db, pending=True)
    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = user.id
    session.email_verified_at = datetime.now(UTC)
    db.session.add(session)
    db.session.commit()

    response = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "Password123!"},
    )
    assert response.status_code == 200, response.get_json()
    db.session.refresh(user)
    db.session.refresh(portal_client)
    assert user.account_status == "active"
    assert user.phone_verified_at is None
    assert portal_client.is_active is True


def test_login_still_blocked_without_email(client, db):
    user, _portal_client = _make_portal_user(db, pending=True)
    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = user.id
    db.session.add(session)
    db.session.commit()

    response = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "Password123!"},
    )
    assert response.status_code == 403
    body = response.get_json() or {}
    assert body.get("reason") == "account_pending_activation"
    db.session.refresh(user)
    assert user.account_status == "pending_activation"
    assert user.phone_verified_at is None


def test_booking_requires_phone_then_succeeds_once_verified(
    client, app, db, monkeypatch
):
    from services.legal.record_terms_acceptance import record_portal_terms_acceptance

    user, portal_client = _make_portal_user(db)
    record_portal_terms_acceptance(user, portal_client)
    db.session.commit()
    headers = _headers(app, user)
    url = f"/api/v1/clients/{user.public_id}/bookings"
    payload = _valid_booking_payload()

    blocked = client.post(url, json=payload, headers=headers)
    assert blocked.status_code == 403, blocked.get_json()
    body = blocked.get_json() or {}
    assert body.get("error") == "phone_verification_required"

    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = user.id
    session.sms_code_hash = auth._hash_plain_value("123456")
    session.sms_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    session.sms_attempts = 0
    db.session.add(session)
    db.session.commit()

    created: list[int] = []
    fake_booking = MagicMock()
    fake_booking.id = 8801
    fake_booking.status = "pending"
    fake_booking.amount = 50.0
    fake_booking.price_amount = 50.0
    fake_booking.price_breakdown_json = {}
    fake_booking.billed_to_type = "patient"

    def _create(**_kwargs):
        created.append(1)
        return fake_booking

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        _create,
    )

    verified = client.post(
        "/api/v1/auth/phone/verify-code",
        json={"code": "123456"},
        headers=headers,
    )
    assert verified.status_code == 200, verified.get_json()

    ok = client.post(url, json=payload, headers=headers)
    assert ok.status_code == 201, ok.get_json()
    assert len(created) == 1


def test_transport_client_booking_does_not_require_sms(
    client, app, db, sample_client, monkeypatch
):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    user.phone_verified_at = None
    db.session.commit()

    fake_booking = MagicMock()
    fake_booking.id = 8802
    fake_booking.status = "pending"
    fake_booking.amount = 50.0
    fake_booking.price_amount = 50.0
    fake_booking.price_breakdown_json = {}
    fake_booking.billed_to_type = "patient"
    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        lambda **_k: fake_booking,
    )
    response = client.post(
        f"/api/v1/clients/{user.public_id}/bookings",
        json=_valid_booking_payload(),
        headers=_headers(app, user),
    )
    assert response.status_code == 201, response.get_json()


def test_phone_send_code_provider_down_keeps_unverified(client, app, db, monkeypatch):
    user, _portal_client = _make_portal_user(db)
    db.session.commit()
    monkeypatch.setattr(
        auth,
        "_send_activation_sms",
        lambda *_a, **_k: {"ok": False, "error_class": "DISABLED", "disabled": True},
    )
    response = client.post(
        "/api/v1/auth/phone/send-code",
        json={},
        headers=_headers(app, user),
    )
    assert response.status_code == 503
    body = response.get_json() or {}
    assert body.get("error") in {"sms_unavailable", "sms_provider_unavailable"}
    db.session.refresh(user)
    assert user.phone_verified_at is None


def test_phone_verify_code_sets_timestamp(client, app, db, monkeypatch):
    user, _portal_client = _make_portal_user(db)
    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = user.id
    session.sms_code_hash = auth._hash_plain_value("123456")
    session.sms_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    session.sms_attempts = 0
    db.session.add(session)
    db.session.commit()

    bad = client.post(
        "/api/v1/auth/phone/verify-code",
        json={"code": "000000"},
        headers=_headers(app, user),
    )
    assert bad.status_code == 400
    db.session.refresh(user)
    assert user.phone_verified_at is None

    ok = client.post(
        "/api/v1/auth/phone/verify-code",
        json={"code": "123456"},
        headers=_headers(app, user),
    )
    assert ok.status_code == 200, ok.get_json()
    db.session.refresh(user)
    assert user.phone_verified_at is not None


def test_phone_change_revokes_verification(client, app, db):
    user, portal_client = _make_portal_user(db)
    user.phone_verified_at = datetime.now(UTC)
    db.session.commit()
    headers = _headers(app, user)

    changed = client.put(
        f"/api/v1/clients/{user.public_id}",
        json={"phone": "+41791230000"},
        headers=headers,
    )
    assert changed.status_code == 200, changed.get_json()
    db.session.refresh(user)
    assert user.phone == "+41791230000"
    assert user.phone_verified_at is None

    user.phone = "+41768190077"
    user.phone_verified_at = datetime.now(UTC)
    db.session.commit()
    keep = client.put(
        f"/api/v1/clients/{user.public_id}",
        json={"phone": "076 819 00 77"},
        headers=headers,
    )
    assert keep.status_code == 200, keep.get_json()
    db.session.refresh(user)
    assert user.phone == "+41768190077"
    assert user.phone_verified_at is not None
    assert portal_client.id is not None


def test_institution_user_is_not_auto_promoted():
    institution = SimpleNamespace(
        role=UserRole.INSTITUTION,
        account_status="pending_activation",
        clients=[],
        phone_verified_at=None,
    )
    from services.auth.portal_phone_verification import maybe_promote_portal_account

    assert maybe_promote_portal_account(institution) is False
    assert institution.account_status == "pending_activation"
    assert institution.phone_verified_at is None


def _fake_booking(booking_id: int):
    fake_booking = MagicMock()
    fake_booking.id = booking_id
    fake_booking.status = "pending"
    fake_booking.amount = 50.0
    fake_booking.price_amount = 50.0
    fake_booking.price_breakdown_json = {}
    fake_booking.billed_to_type = "patient"
    return fake_booking


def test_verified_account_books_twice_without_sms(client, app, db, monkeypatch):
    from services.legal.record_terms_acceptance import record_portal_terms_acceptance

    user, portal_client = _make_portal_user(db)
    user.phone_verified_at = datetime.now(UTC)
    record_portal_terms_acceptance(user, portal_client)
    db.session.commit()
    sms_calls: list[str] = []

    def _forbid_sms(*_args, **_kwargs):
        sms_calls.append("sms")
        return {"ok": True}

    monkeypatch.setattr(auth, "_send_activation_sms", _forbid_sms)
    monkeypatch.setattr("services.notifications.sms.send_sms_notification", _forbid_sms)
    created: list[int] = []

    def _create(**_kwargs):
        created.append(1)
        return _fake_booking(8900 + len(created))

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        _create,
    )
    headers = _headers(app, user)
    payload = _valid_booking_payload()
    urls = [
        f"/api/v1/clients/{user.public_id}/bookings",
        "/api/v1/clients/me/bookings",
        f"/api/v1/bookings/clients/{user.public_id}/bookings",
    ]
    for url in urls[:2]:
        response = client.post(url, json=payload, headers=headers)
        assert response.status_code == 201, response.get_json()
    third = client.post(urls[2], json=payload, headers=headers)
    assert third.status_code == 201, third.get_json()
    assert len(created) == 3
    assert sms_calls == []


def test_phone_change_blocks_booking_until_the_new_number_is_verified(
    client, app, db, monkeypatch
):
    from services.legal.record_terms_acceptance import record_portal_terms_acceptance

    user, portal_client = _make_portal_user(db)
    user.phone_verified_at = datetime.now(UTC)
    record_portal_terms_acceptance(user, portal_client)
    db.session.commit()
    headers = _headers(app, user)
    changed = client.put(
        f"/api/v1/clients/{user.public_id}",
        json={"phone": "+41790001122"},
        headers=headers,
    )
    assert changed.status_code == 200, changed.get_json()
    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        lambda **_k: _fake_booking(8910),
    )
    blocked = client.post(
        f"/api/v1/clients/{user.public_id}/bookings",
        json=_valid_booking_payload(),
        headers=headers,
    )
    assert blocked.status_code == 403
    assert (blocked.get_json() or {}).get("error") == "phone_verification_required"

    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = user.id
    session.sms_code_hash = auth._hash_plain_value("654321")
    session.sms_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    session.sms_attempts = 0
    db.session.add(session)
    db.session.commit()
    verified = client.post(
        "/api/v1/auth/phone/verify-code",
        json={"code": "654321"},
        headers=headers,
    )
    assert verified.status_code == 200, verified.get_json()
    ok = client.post(
        f"/api/v1/clients/{user.public_id}/bookings",
        json=_valid_booking_payload(),
        headers=headers,
    )
    assert ok.status_code == 201, ok.get_json()
    again = client.post(
        "/api/v1/clients/me/bookings",
        json=_valid_booking_payload(),
        headers=headers,
    )
    assert again.status_code == 201, again.get_json()

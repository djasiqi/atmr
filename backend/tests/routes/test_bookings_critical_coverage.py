"""Couverture critique ``routes/bookings.py`` (seuil 80 %)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

from flask_jwt_extended import create_access_token

from models import Booking, Client, User
from models.enums import BookingStatus, UserRole
from routes import bookings as bookings_mod


def _headers(app, user: User, *, role: str | None = None) -> dict[str, str]:
    claims = {
        "role": role or str(getattr(user.role, "value", user.role)),
        "aud": "atmr-api",
    }
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _make_user(db, role: UserRole, *, suffix: str | None = None) -> User:
    suffix = suffix or uuid.uuid4().hex[:8]
    user = User()
    user.username = f"bk_{role.value}_{suffix}"
    user.email = f"bk_{role.value}_{suffix}@test.ch"
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    return user


def _make_client(db, user: User, company_id: int) -> Client:
    client = Client()
    client.user_id = user.id
    client.company_id = company_id
    client.contact_email = user.email
    client.is_active = True
    db.session.add(client)
    db.session.flush()
    return client


def _make_booking(
    db,
    *,
    company_id: int,
    client_id: int,
    user_id: int,
    status: BookingStatus = BookingStatus.PENDING,
    driver_id: int | None = None,
    scheduled_time: datetime | None = None,
) -> Booking:
    booking = Booking()
    booking.company_id = company_id
    booking.client_id = client_id
    booking.user_id = user_id
    booking.customer_name = "Couverture"
    booking.pickup_location = "Rue A 1, Genève"
    booking.dropoff_location = "Rue B 2, Genève"
    booking.scheduled_time = scheduled_time or (datetime.now(UTC) + timedelta(hours=4))
    booking.status = status
    booking.amount = Decimal("40.00")
    if driver_id is not None:
        booking.driver_id = driver_id
    db.session.add(booking)
    db.session.commit()
    return booking


# ---------------------------------------------------------------------------
# Helpers unitaires
# ---------------------------------------------------------------------------


def test_queue_trigger_skip_and_apis(monkeypatch):
    bookings_mod._queue_trigger(None, "update")

    calls: list[tuple] = []

    def trigger_on_booking_change(cid, reason=None, origin=None):
        calls.append(("modern", cid, reason, origin))

    monkeypatch.setattr(
        bookings_mod.queue, "trigger_on_booking_change", trigger_on_booking_change
    )
    bookings_mod._queue_trigger(7, "cancel")
    assert calls[0][0] == "modern"
    assert calls[0][2] == "booking_cancel"

    monkeypatch.setattr(bookings_mod.queue, "trigger_on_booking_change", None)

    def trigger(cid, reason=None, mode=None, origin=None):
        calls.append(("alt", cid, reason, mode, origin))

    monkeypatch.setattr(bookings_mod.queue, "trigger", trigger)
    bookings_mod._queue_trigger(8, "update")
    assert calls[-1][0] == "alt"

    def boom(*_a, **_k):
        raise RuntimeError("queue down")

    monkeypatch.setattr(bookings_mod.queue, "trigger_on_booking_change", boom)
    bookings_mod._queue_trigger(9, "update")


def test_build_pagination_links(app):
    with app.test_request_context("/"):
        app.config["SERVER_NAME"] = None
        app.config["PREFERRED_URL_SCHEME"] = "https"
        headers = bookings_mod._build_pagination_links(
            2, 10, 35, "bookings_list_bookings"
        )
        assert 'rel="prev"' in headers["Link"]
        assert 'rel="next"' in headers["Link"]
        assert headers["X-Total-Count"] == "35"
        assert headers["X-Page"] == "2"


def test_check_booking_ownership_roles(monkeypatch):
    booking = SimpleNamespace(id=1, company_id=10, client_id=20, driver_id=30)

    admin = SimpleNamespace(id=1, role=UserRole.admin, public_id="a")
    ok, err = bookings_mod._check_booking_ownership(booking, admin, "modify")
    assert ok is True
    assert err is None

    company_user = SimpleNamespace(id=2, role=UserRole.company, public_id="c")
    company_repo = MagicMock()
    company_repo.find_by_user_id.return_value = SimpleNamespace(id=10)
    monkeypatch.setattr(
        "repositories.company_repository.CompanyRepository",
        lambda: company_repo,
    )
    ok, err = bookings_mod._check_booking_ownership(booking, company_user, "read")
    assert ok is True
    company_repo.find_by_user_id.return_value = SimpleNamespace(id=99)
    ok, err = bookings_mod._check_booking_ownership(booking, company_user, "read")
    assert ok is False
    assert err is not None

    client_user = SimpleNamespace(id=3, role=UserRole.client, public_id="cl")
    monkeypatch.setattr(bookings_mod.client_repo, "find_by_user_id", lambda _uid: None)
    ok, err = bookings_mod._check_booking_ownership(booking, client_user, "read")
    assert ok is False

    monkeypatch.setattr(
        bookings_mod.client_repo,
        "find_by_user_id",
        lambda _uid: SimpleNamespace(id=20),
    )
    ok, err = bookings_mod._check_booking_ownership(booking, client_user, "read")
    assert ok is True

    monkeypatch.setattr(
        bookings_mod.client_repo,
        "find_by_user_id",
        lambda _uid: SimpleNamespace(id=21),
    )
    ok, err = bookings_mod._check_booking_ownership(booking, client_user, "modify")
    assert ok is False

    driver_user = SimpleNamespace(id=4, role=UserRole.driver, public_id="d")
    monkeypatch.setattr(
        bookings_mod.driver_repo,
        "find_model_by_user_id",
        lambda _uid: SimpleNamespace(id=30),
    )
    ok, err = bookings_mod._check_booking_ownership(booking, driver_user, "read")
    assert ok is True
    ok, err = bookings_mod._check_booking_ownership(booking, driver_user, "modify")
    assert ok is False


def test_geocoding_and_iso_and_period_and_status():
    body, code = bookings_mod._handle_geocoding_error(
        RuntimeError("Service temporairement indisponible")
    )
    assert code == 400
    assert body["error"] == "erreur_geocodage"
    body, code = bookings_mod._handle_geocoding_error(RuntimeError("adresse inconnue"))
    assert body["error"] == "impossible_de_geocoder"

    assert bookings_mod._parse_iso_or_none(None) is None
    assert bookings_mod._parse_iso_or_none("not-a-date") is None
    parsed = bookings_mod._parse_iso_or_none("2026-08-13T10:00:00Z")
    assert parsed is not None
    assert parsed.tzinfo is not None

    this_month = bookings_mod._period_bounds("this_month", None, None)
    assert this_month is not None
    prev = bookings_mod._period_bounds("previous_month", None, None)
    assert prev is not None
    assert prev[0] < prev[1]
    year = bookings_mod._period_bounds("this_year", None, None)
    assert year is not None
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 2, 1, tzinfo=UTC)
    custom = bookings_mod._period_bounds("custom", start, end)
    assert custom == (start, end)
    assert bookings_mod._period_bounds("custom", end, start) is None
    assert bookings_mod._period_bounds("unknown", None, None) is None

    # Décembre : branche année suivante
    frozen = datetime(2026, 12, 15, tzinfo=UTC)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    original = bookings_mod.datetime
    bookings_mod.datetime = _FrozenDateTime  # type: ignore[misc]
    try:
        dec = bookings_mod._period_bounds("this_month", None, None)
        assert dec is not None
        assert dec[1].year == 2027
    finally:
        bookings_mod.datetime = original  # type: ignore[misc]

    assert bookings_mod._status_label("COMPLETED") == "Terminée"
    assert bookings_mod._status_label("assigned") == "Confirmée"
    assert bookings_mod._status_label("en_route") == "Chauffeur en route"
    assert bookings_mod._status_label("weird") == "En attente"


def test_build_client_bookings_pdf_and_page_break():
    rows = []
    for i in range(40):
        rows.append(
            SimpleNamespace(
                id=i + 1,
                scheduled_time=datetime(2026, 8, 1, 10, i % 50, tzinfo=UTC)
                if i % 2 == 0
                else None,
                amount=10 + i,
                pickup_location="A",
                dropoff_location="B",
                status="completed",
                company=SimpleNamespace(name="Co"),
            )
        )
    pdf = bookings_mod._build_client_bookings_pdf(rows, "Ce mois")
    assert pdf.startswith(b"%PDF")


def test_booking_creation_client_brief_pricing_branches():
    booking = SimpleNamespace(
        id=1,
        status=BookingStatus.PENDING,
        amount=50,
        price_amount=45,
        price_breakdown_json={
            "overridden_by_preferential": True,
            "preferential_source": "clinic",
        },
        billed_to_type="patient",
    )
    brief = bookings_mod._booking_creation_client_brief(booking)
    assert brief["pricing_status"] == "adjusted"
    assert "clinique" in (brief["pricing_adjustment_reason"] or "")

    booking.price_breakdown_json = {
        "overridden_by_preferential": True,
        "preferential_source": "client",
    }
    assert "client" in (
        bookings_mod._booking_creation_client_brief(booking)[
            "pricing_adjustment_reason"
        ]
        or ""
    )

    booking.price_breakdown_json = {"overridden_by_preferential": True}
    assert "règle métier" in (
        bookings_mod._booking_creation_client_brief(booking)[
            "pricing_adjustment_reason"
        ]
        or ""
    )

    booking.price_breakdown_json = {"pricing_amount_applied": True}
    brief = bookings_mod._booking_creation_client_brief(booking)
    assert brief["pricing_status"] == "adjusted"


def test_validate_user_and_client(monkeypatch):
    monkeypatch.setattr(
        "shared.infrastructure.adapters.auth_adapter.get_current_user_via_use_case",
        lambda: None,
    )
    _user, _client, err = bookings_mod._validate_user_and_client("abc")
    assert err is not None
    assert err[1] == 401

    current = SimpleNamespace(id=11)
    monkeypatch.setattr(
        "shared.infrastructure.adapters.auth_adapter.get_current_user_via_use_case",
        lambda: current,
    )
    monkeypatch.setattr(
        bookings_mod.client_repo, "find_by_public_id", lambda _pid: None
    )
    _user, _client, err = bookings_mod._validate_user_and_client("abc")
    assert err is not None
    assert err[1] == 403

    monkeypatch.setattr(
        bookings_mod.client_repo,
        "find_by_public_id",
        lambda _pid: SimpleNamespace(user_id=11),
    )
    user, _client, err = bookings_mod._validate_user_and_client("abc")
    assert err is None
    assert user is current


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def test_export_pdf_period_validation_and_success(
    client, app, db, sample_company, sample_client
):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    _make_booking(
        db,
        company_id=sample_company.id,
        client_id=sample_client.id,
        user_id=user.id,
        scheduled_time=datetime.now(UTC),
    )
    headers = _headers(app, user, role="client")

    bad = client.post(
        "/api/v1/bookings/clients/me/bookings/export-pdf",
        headers=headers,
        json={"period": "nope"},
    )
    assert bad.status_code == 400

    custom_bad = client.post(
        "/api/v1/bookings/clients/me/bookings/export-pdf",
        headers=headers,
        json={
            "period": "custom",
            "from": "2026-02-01T00:00:00Z",
            "to": "2026-01-01T00:00:00Z",
        },
    )
    assert custom_bad.status_code == 400

    ok = client.post(
        "/api/v1/bookings/clients/me/bookings/export-pdf",
        headers=headers,
        json={"period": "this_year"},
    )
    assert ok.status_code == 200
    assert ok.mimetype == "application/pdf"


def test_saferpay_initialize_and_assert(
    client, app, db, sample_company, sample_client, monkeypatch
):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    booking = _make_booking(
        db,
        company_id=sample_company.id,
        client_id=sample_client.id,
        user_id=user.id,
        status=BookingStatus.AWAITING_CLIENT_PAYMENT,
    )
    headers = _headers(app, user, role="client")

    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: False)
    monkeypatch.setattr("routes.bookings.saferpay_configured", lambda: False)
    init = client.post(
        f"/api/v1/bookings/{booking.id}/saferpay/initialize",
        headers=headers,
        json={},
    )
    assert init.status_code == 503

    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: True)
    monkeypatch.setattr("routes.bookings.saferpay_configured", lambda: True)
    monkeypatch.setattr(
        "routes.bookings.create_saferpay_payment_page_initialize",
        lambda **_k: {"redirect_url": "https://pay.test"},
    )
    init_ok = client.post(
        f"/api/v1/bookings/{booking.id}/saferpay/initialize",
        headers=headers,
        json={"return_url": "https://app.test/return"},
    )
    assert init_ok.status_code == 200

    assert_missing = client.post(
        f"/api/v1/bookings/{booking.id}/saferpay/assert",
        headers=headers,
        json={},
    )
    assert assert_missing.status_code == 400

    assert_bad = client.post(
        f"/api/v1/bookings/{booking.id}/saferpay/assert",
        headers=headers,
        json={"payment_id": "abc"},
    )
    assert assert_bad.status_code == 400


def test_get_put_delete_booking(client, app, db, sample_company, sample_client):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    booking = _make_booking(
        db,
        company_id=sample_company.id,
        client_id=sample_client.id,
        user_id=user.id,
    )
    headers = _headers(app, user, role="client")

    get_resp = client.get(f"/api/v1/bookings/{booking.id}", headers=headers)
    assert get_resp.status_code == 200

    put_resp = client.put(
        f"/api/v1/bookings/{booking.id}",
        headers=headers,
        json={"notes_medical": "note couverture"},
    )
    assert put_resp.status_code in (200, 400)

    delete_resp = client.delete(f"/api/v1/bookings/{booking.id}", headers=headers)
    assert delete_resp.status_code in (200, 400)


def test_list_bookings_client_and_admin(
    client,
    app,
    db,
    sample_company,
    sample_client,
    sample_admin_user,
    sample_user,
    monkeypatch,
):
    monkeypatch.setattr(
        bookings_mod,
        "_build_pagination_links",
        lambda *_a, **_k: {
            "Link": "",
            "X-Total-Count": "1",
            "X-Page": "1",
            "X-Per-Page": "10",
            "X-Total-Pages": "1",
        },
    )
    client_user = db.session.get(User, sample_client.user_id)
    assert client_user is not None
    _make_booking(
        db,
        company_id=sample_company.id,
        client_id=sample_client.id,
        user_id=client_user.id,
    )
    client_headers = _headers(app, client_user, role="client")
    listed = client.get("/api/v1/bookings/?page=1&per_page=10", headers=client_headers)
    assert listed.status_code == 200
    body = listed.get_json()
    assert "bookings" in body

    admin_headers = _headers(app, sample_admin_user, role="admin")
    admin_list = client.get(
        "/api/v1/bookings/?page=1&per_page=10",
        headers=admin_headers,
    )
    assert admin_list.status_code == 200

    company_headers = _headers(app, sample_user, role="company")
    forbidden = client.get("/api/v1/bookings/", headers=company_headers)
    assert forbidden.status_code in (403, 401)


def test_create_booking_validation_and_idempotency(
    client, app, db, sample_company, sample_client, monkeypatch
):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    headers = _headers(app, user, role="client")
    url = f"/api/v1/bookings/clients/{user.public_id}/bookings"

    bad = client.post(url, headers=headers, json={})
    assert bad.status_code == 400

    fake_booking = SimpleNamespace(
        id=999,
        status=BookingStatus.PENDING,
        amount=50,
        price_amount=50,
        price_breakdown_json=None,
        billed_to_type="patient",
        company_id=sample_company.id,
    )
    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        lambda **_k: fake_booking,
    )
    scheduled = (datetime.now(UTC) + timedelta(hours=5)).replace(microsecond=0)
    iso = scheduled.isoformat().replace("+00:00", "Z")
    ok = client.post(
        url,
        headers=headers,
        json={
            "customer_name": "Test",
            "pickup_location": "Rue Pickup 1, Genève",
            "dropoff_location": "Rue Dropoff 2, Genève",
            "scheduled_time": iso,
            "amount": 50.0,
        },
    )
    assert ok.status_code in (200, 201), ok.get_json()

    monkeypatch.setattr(
        "services.security.idempotency.IdempotencyService.get_idempotency_key_from_request",
        lambda: "idem-1",
    )
    monkeypatch.setattr(
        "services.security.idempotency.IdempotencyService.check_key",
        lambda _k: (True, {"response": {"cached": True}, "status_code": 201}),
    )
    cached = client.post(
        url,
        headers={**headers, "Idempotency-Key": "idem-1"},
        json={
            "customer_name": "Test",
            "pickup_location": "Rue Pickup 1, Genève",
            "dropoff_location": "Rue Dropoff 2, Genève",
            "scheduled_time": iso,
            "amount": 50.0,
        },
    )
    assert cached.status_code == 201


def test_create_booking_error_branches(
    client, app, db, sample_company, sample_client, monkeypatch
):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    headers = _headers(app, user, role="client")
    url = f"/api/v1/bookings/clients/{user.public_id}/bookings"
    scheduled = (datetime.now(UTC) + timedelta(hours=5)).replace(microsecond=0)
    iso = scheduled.isoformat().replace("+00:00", "Z")
    payload = {
        "customer_name": "Test",
        "pickup_location": "Rue Pickup 1, Genève",
        "dropoff_location": "Rue Dropoff 2, Genève",
        "scheduled_time": iso,
        "amount": 50.0,
    }

    from application.bookings.create_booking import InvalidClientBookingCommand

    def raise_invalid(**_k):
        raise InvalidClientBookingCommand(["amount_source"])

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        raise_invalid,
    )
    invalid = client.post(url, headers=headers, json=payload)
    assert invalid.status_code == 400

    def raise_value(**_k):
        raise ValueError("données invalides")

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        raise_value,
    )
    value_err = client.post(url, headers=headers, json=payload)
    assert value_err.status_code == 400

    def raise_geo(**_k):
        raise RuntimeError("temporairement indisponible")

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        raise_geo,
    )
    geo = client.post(url, headers=headers, json=payload)
    assert geo.status_code == 400

    from services.platform_exceptions import PlatformTenantSuspended

    def raise_suspended(**_k):
        raise PlatformTenantSuspended("suspendu")

    monkeypatch.setattr(
        "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
        raise_suspended,
    )
    suspended = client.post(url, headers=headers, json=payload)
    assert suspended.status_code == 403


def test_get_booking_404_and_idor(client, app, db, sample_company, sample_client):
    user = db.session.get(User, sample_client.user_id)
    assert user is not None
    headers = _headers(app, user, role="client")
    missing = client.get("/api/v1/bookings/999999", headers=headers)
    assert missing.status_code == 404

    other = _make_user(db, UserRole.client, suffix=uuid.uuid4().hex[:8])
    other_client = _make_client(db, other, sample_company.id)
    booking = _make_booking(
        db,
        company_id=sample_company.id,
        client_id=other_client.id,
        user_id=other.id,
    )
    idor = client.get(f"/api/v1/bookings/{booking.id}", headers=headers)
    assert idor.status_code == 403

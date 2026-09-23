"""Réacceptation des conditions PORTAL : statut, gate et registre append-only."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from application.bookings.cancel_booking import CancelBookingInput, CancelBookingUseCase
from domain.bookings.commands import CreateBookingCommand
from models.booking import Booking
from models.client_booking_contract_event import ClientBookingContractEvent
from models.client_terms_acceptance import (
    DOCUMENT_TERMS_OF_SERVICE,
    DOCUMENT_TRANSPORT_TERMS,
    VERIFICATION_NOT_VERIFIED,
    VERIFICATION_OTP_SMS,
    ClientTermsAcceptance,
)
from models.enums import BookingStatus, ClientType
from models.user import User
from services.legal.portal_terms_catalog import (
    PublishedTerms,
    canonical_sha256,
    current_portal_terms,
)
from services.legal.portal_terms_status import (
    BASIS_CURRENT_ACCEPTANCE,
    BASIS_PRIOR_ACCEPTANCE,
    STATUS_CURRENT,
    STATUS_REACCEPTANCE_REQUIRED,
    PortalTermsReacceptanceRequired,
    accept_current_required_portal_terms,
    resolve_portal_terms_status,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance
from tests.helpers.create_booking_use_case import CreateBookingUseCase
from tests.routes.test_auth_sms_02_portal_contract import (
    _headers,
    _make_portal_user,
    _valid_booking_payload,
)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _names(user: User) -> None:
    user.first_name = "Jeanne"
    user.last_name = "Martin"


def _booking(user, client) -> Booking:
    booking = Booking()
    booking.customer_name = "Jeanne Martin"
    booking.pickup_location = "Rue du Port 1, Genève"
    booking.dropoff_location = "HUG, Genève"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=2)
    booking.amount = 90.0
    booking.status = BookingStatus.PENDING
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.is_round_trip = False
    return booking


def _spec(
    document_type: str, version: str, body: str, *, required: bool
) -> PublishedTerms:
    return PublishedTerms(
        document_type=document_type,
        terms_version=version,
        terms_hash=canonical_sha256(body),
        canonical_body=body,
        requires_reacceptance=required,
    )


def _publish(monkeypatch: pytest.MonkeyPatch, *specs: PublishedTerms) -> None:
    monkeypatch.setattr(
        "services.legal.portal_terms_catalog.current_portal_terms",
        lambda: tuple(specs),
    )


def _post_accept(client, app, user, **extra):
    return client.post(
        "/api/v1/clients/me/terms-acceptances",
        json={"accept_current_required_terms": True, **extra},
        headers=_headers(app, user),
    )


def _post_booking(client, app, user, monkeypatch=None):
    if monkeypatch is not None:
        fake = MagicMock()
        fake.id = 8801
        fake.status = "pending"
        fake.amount = 50.0
        fake.price_amount = 50.0
        fake.price_breakdown_json = {}
        fake.billed_to_type = "patient"
        fake.company_id = None
        monkeypatch.setattr(
            "bookings.infrastructure.adapters.booking_service_adapter.create_booking_via_use_case",
            lambda **_k: fake,
        )
    return client.post(
        f"/api/v1/clients/{user.public_id}/bookings",
        json=_valid_booking_payload(),
        headers=_headers(app, user),
    )


def test_current_acceptance_allows_booking_and_matches_catalog_hash(
    client, app, db, monkeypatch
) -> None:
    user, portal = _make_portal_user(db)
    user.phone_verified_at = datetime.now(UTC)
    record_portal_terms_acceptance(user, portal)
    db.session.commit()

    resolved = resolve_portal_terms_status(user)
    assert resolved.status == STATUS_CURRENT
    catalog = {spec.document_type: spec for spec in current_portal_terms()}
    for document in resolved.documents:
        spec = catalog[document.document_type]
        assert document.acceptance_required is False
        assert document.contractual_basis == BASIS_CURRENT_ACCEPTANCE
        assert document.current_version == "1.0"
        assert document.current_hash == spec.terms_hash
        assert document.current_hash == canonical_sha256(spec.canonical_body)
        row = db.session.get(ClientTermsAcceptance, document.acceptance_id)
        assert row is not None
        assert row.terms_hash == document.current_hash

    response = _post_booking(client, app, user, monkeypatch)
    assert response.status_code == 201, response.get_json()


def test_historical_account_blocks_direct_booking(client, app, db) -> None:
    user, _portal = _make_portal_user(db)
    user.phone_verified_at = datetime.now(UTC)
    db.session.commit()

    login = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "Password123!"},
    )
    assert login.status_code == 200, login.get_json()

    status = client.get(
        "/api/v1/clients/me/portal-terms-status",
        headers=_headers(app, user),
    )
    assert status.status_code == 200, status.get_json()
    body = status.get_json()["data"]
    assert body["status"] == STATUS_REACCEPTANCE_REQUIRED
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0

    blocked = _post_booking(client, app, user)
    assert blocked.status_code == 403, blocked.get_json()
    payload = blocked.get_json()
    assert payload["error"] == "terms_reacceptance_required"
    assert set(payload["details"]["documents"]) == {
        DOCUMENT_TERMS_OF_SERVICE,
        DOCUMENT_TRANSPORT_TERMS,
    }
    assert Booking.query.filter_by(user_id=user.id).count() == 0
    assert (
        ClientBookingContractEvent.query.filter_by(actor_user_id=user.id).count() == 0
    )


def test_unchecked_or_forged_version_does_not_accept(client, app, db) -> None:
    user, _portal = _make_portal_user(db)
    db.session.commit()
    headers = _headers(app, user)
    missing = client.post(
        "/api/v1/clients/me/terms-acceptances",
        json={},
        headers=headers,
    )
    unchecked = client.post(
        "/api/v1/clients/me/terms-acceptances",
        json={"accept_current_required_terms": False},
        headers=headers,
    )
    forged = client.post(
        "/api/v1/clients/me/terms-acceptances",
        json={
            "accept_current_required_terms": True,
            "terms_version": "9.9",
            "terms_hash": "ab" * 32,
        },
        headers=headers,
    )
    assert missing.status_code == 400
    assert missing.get_json()["error"] == "terms_acceptance_required"
    assert unchecked.status_code == 400
    assert forged.status_code == 400
    assert forged.get_json()["error"] == "client_supplied_terms_forbidden"
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0


def test_repeat_accept_does_not_duplicate_and_second_insert_rolls_back(
    client, app, db, monkeypatch
) -> None:
    user, _portal = _make_portal_user(db)
    db.session.commit()
    first = _post_accept(client, app, user)
    second = _post_accept(client, app, user)
    assert first.status_code == 201, first.get_json()
    assert second.status_code == 200, second.get_json()
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 2
    assert "with_for_update" in inspect.getsource(accept_current_required_portal_terms)

    other, _other_client = _make_portal_user(db)
    db.session.commit()

    def fail_second(portal_user, portal_client, documents=None):
        specs = list(documents or [])
        record_portal_terms_acceptance(portal_user, portal_client, documents=specs[:1])
        raise RuntimeError("deuxième insertion interrompue")

    monkeypatch.setattr(
        "services.legal.portal_terms_status.record_portal_terms_acceptance",
        fail_second,
    )
    failed = _post_accept(client, app, other)
    assert failed.status_code == 500
    db.session.expire_all()
    assert ClientTermsAcceptance.query.filter_by(user_id=other.id).count() == 0
    fresh = db.session.get(User, other.id)
    assert fresh is not None
    assert resolve_portal_terms_status(fresh).status == STATUS_REACCEPTANCE_REQUIRED


def test_substantial_transport_version_then_reacceptance_relinks_new_booking(
    client, app, db, monkeypatch
) -> None:
    user, portal = _make_portal_user(db)
    _names(user)
    user.phone_verified_at = None
    rows = record_portal_terms_acceptance(user, portal)
    cgu_v1 = next(row for row in rows if row.document_type == DOCUMENT_TERMS_OF_SERVICE)
    transport_v1 = next(
        row for row in rows if row.document_type == DOCUMENT_TRANSPORT_TERMS
    )
    assert transport_v1.verification_method == VERIFICATION_NOT_VERIFIED
    old_booking = _booking(user, portal)
    db.session.add(old_booking)
    db.session.flush()
    old_event = record_portal_booking_created_event(
        booking=old_booking, user_id=user.id
    )
    db.session.commit()
    old_event_id = old_event.id
    old_transport_acceptance_id = old_event.transport_terms_acceptance_id

    cgu, _transport = current_portal_terms()
    transport_v2 = _spec(
        DOCUMENT_TRANSPORT_TERMS,
        "2.0",
        "LIRIE — CGV transport\nterms_version: 2.0\nVersion substantielle de test.\n",
        required=True,
    )
    _publish(monkeypatch, cgu, transport_v2)

    stale = resolve_portal_terms_status(user)
    assert stale.status == STATUS_REACCEPTANCE_REQUIRED
    required = [doc for doc in stale.documents if doc.acceptance_required]
    assert [doc.document_type for doc in required] == [DOCUMENT_TRANSPORT_TERMS]
    assert required[0].current_hash == transport_v2.terms_hash

    blocked = _post_booking(client, app, user)
    assert blocked.status_code == 403
    assert Booking.query.filter_by(user_id=user.id).count() == 1

    cancel = CancelBookingUseCase().execute(CancelBookingInput(booking=old_booking))
    assert cancel.success is True
    db.session.commit()

    user.phone_verified_at = datetime.now(UTC)
    db.session.commit()
    accepted = _post_accept(client, app, user)
    assert accepted.status_code == 201, accepted.get_json()
    replay = _post_accept(client, app, user)
    assert replay.status_code == 200
    stored = ClientTermsAcceptance.query.filter_by(user_id=user.id).all()
    assert len(stored) == 3
    transport_rows = sorted(
        [row for row in stored if row.document_type == DOCUMENT_TRANSPORT_TERMS],
        key=lambda row: int(row.id),
    )
    assert [row.terms_version for row in transport_rows] == ["1.0", "2.0"]
    fresh_v1 = db.session.get(ClientTermsAcceptance, transport_v1.id)
    assert fresh_v1 is not None
    assert fresh_v1.terms_version == "1.0"
    assert fresh_v1.verification_method == VERIFICATION_NOT_VERIFIED
    assert fresh_v1.phone_verified_at_snapshot is None
    new_transport = next(row for row in transport_rows if row.terms_version == "2.0")
    assert new_transport.terms_hash == transport_v2.terms_hash
    assert new_transport.verification_method == VERIFICATION_OTP_SMS
    assert new_transport.phone_verified_at_snapshot is not None
    assert db.session.get(ClientTermsAcceptance, cgu_v1.id).terms_version == "1.0"

    current = resolve_portal_terms_status(user)
    assert current.status == STATUS_CURRENT
    new_booking = _booking(user, portal)
    new_booking.customer_name = "Jeanne Martin retour"
    db.session.add(new_booking)
    db.session.flush()
    new_event = record_portal_booking_created_event(
        booking=new_booking, user_id=user.id
    )
    kept = db.session.get(ClientBookingContractEvent, old_event_id)
    assert kept is not None
    assert kept.transport_terms_acceptance_id == old_transport_acceptance_id
    assert kept.terms_of_service_acceptance_id == cgu_v1.id
    assert new_event.transport_terms_acceptance_id == new_transport.id
    assert new_event.terms_of_service_acceptance_id == cgu_v1.id
    assert new_event.amount_is_contractual is False


def test_non_substantial_version_keeps_prior_acceptance(db, monkeypatch) -> None:
    user, portal = _make_portal_user(db)
    _names(user)
    rows = record_portal_terms_acceptance(user, portal)
    cgu_v1 = next(row for row in rows if row.document_type == DOCUMENT_TERMS_OF_SERVICE)
    _cgu, transport = current_portal_terms()
    cgu_v11 = _spec(
        DOCUMENT_TERMS_OF_SERVICE,
        "1.1",
        "LIRIE — CGU\nterms_version: 1.1\nModification informative de test.\n",
        required=False,
    )
    _publish(monkeypatch, cgu_v11, transport)
    resolved = resolve_portal_terms_status(user)
    assert resolved.status == STATUS_CURRENT
    cgu_status = next(
        doc
        for doc in resolved.documents
        if doc.document_type == DOCUMENT_TERMS_OF_SERVICE
    )
    assert cgu_status.acceptance_required is False
    assert cgu_status.requires_reacceptance is False
    assert cgu_status.current_version == "1.1"
    assert cgu_status.accepted_version == "1.0"
    assert cgu_status.acceptance_id == cgu_v1.id
    assert cgu_status.contractual_basis == BASIS_PRIOR_ACCEPTANCE
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 2

    booking = _booking(user, portal)
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.terms_of_service_acceptance_id == cgu_v1.id
    linked = db.session.get(ClientTermsAcceptance, event.terms_of_service_acceptance_id)
    assert linked is not None
    assert linked.terms_version == "1.0"


def test_use_case_blocks_before_writer(db) -> None:
    user, _portal = _make_portal_user(db)
    db.session.commit()
    writer_calls = {"n": 0}

    class _Writer:
        def create_and_commit(self, **_kwargs):
            writer_calls["n"] += 1
            raise AssertionError("une réservation a été écrite")

    class _Repo:
        def find_by_id(self, _client_id: int):
            return SimpleNamespace(
                id=1,
                company_id=None,
                client_type=ClientType.PORTAL,
                preferential_rate=None,
            )

    uc = CreateBookingUseCase(
        client_repo=_Repo(),  # type: ignore[arg-type]
        company_lookup=SimpleNamespace(find_model_by_id=lambda _cid: None),  # type: ignore[arg-type]
        booking_writer=_Writer(),  # type: ignore[arg-type]
        geocoding_service=SimpleNamespace(  # type: ignore[arg-type]
            geocode_address=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("géocodage atteint")
            )
        ),
        distance_duration_fn=lambda _p, _d: (60, 1000),
        company_creation_gate_fn=lambda _cid: None,
        billing_capability_gate_fn=lambda _cid: None,
        fallback_coords_fn=lambda _company: (46.2, 6.1),
    )
    with pytest.raises(PortalTermsReacceptanceRequired):
        uc.execute(
            CreateBookingCommand(
                user_id=user.id,
                client_id=1,
                data=_valid_booking_payload(),
            )
        )
    assert writer_calls["n"] == 0


def test_gate_order_and_existing_booking_flows_are_unchanged() -> None:
    create_source = (
        BACKEND_ROOT / "application" / "bookings" / "create_booking.py"
    ).read_text(encoding="utf-8")
    route_source = (BACKEND_ROOT / "routes" / "bookings.py").read_text(encoding="utf-8")
    execute_uc = create_source.split("def execute(self, cmd: CreateBookingCommand)", 1)[
        1
    ]
    assert execute_uc.index("assert_portal_terms_current") < execute_uc.index(
        "assert_portal_phone_verified"
    )
    assert execute_uc.index("assert_portal_phone_verified") < execute_uc.index(
        "self.booking_writer.create_and_commit"
    )
    execute = route_source.split("def execute_client_booking_creation", 1)[1].split(
        "\ndef ", 1
    )[0]
    assert execute.index("assert_portal_terms_current(") < execute.index(
        "not user_phone_is_verified"
    )
    cancel_source = (
        BACKEND_ROOT / "application" / "bookings" / "cancel_booking.py"
    ).read_text(encoding="utf-8")
    update_source = (
        BACKEND_ROOT / "application" / "bookings" / "update_pending_booking.py"
    ).read_text(encoding="utf-8")
    assert "assert_portal_terms_current" not in cancel_source
    assert "assert_portal_terms_current" not in update_source
    phone_source = (
        BACKEND_ROOT / "services" / "auth" / "portal_phone_verification.py"
    ).read_text(encoding="utf-8")
    assert "terms_reacceptance_required" not in phone_source
    assert "def assert_portal_phone_verified" in phone_source


def test_published_version_flag_cannot_be_rewritten(db) -> None:
    user, portal = _make_portal_user(db)
    record_portal_terms_acceptance(user, portal)
    db.session.commit()
    cgu, transport = current_portal_terms()
    rewritten = PublishedTerms(
        document_type=cgu.document_type,
        terms_version=cgu.terms_version,
        terms_hash=cgu.terms_hash,
        canonical_body=cgu.canonical_body,
        requires_reacceptance=False,
    )
    from services.legal.portal_terms_catalog import CatalogIntegrityError
    from services.legal.record_terms_acceptance import ensure_document_version

    with pytest.raises(CatalogIntegrityError):
        ensure_document_version(rewritten)
    _ = transport

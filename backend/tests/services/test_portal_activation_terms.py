"""Acceptation initiale des CGU et CGV à la finalisation PORTAL."""

from __future__ import annotations

import inspect
import uuid
from datetime import UTC, datetime

from models import ActivationSession, User
from models.client_terms_acceptance import (
    VERIFICATION_NOT_VERIFIED,
    VERIFICATION_OTP_SMS,
    ClientTermsAcceptance,
)
from routes import auth
from services.legal.portal_terms_catalog import current_portal_terms
from services.legal.record_terms_acceptance import record_portal_terms_acceptance
from tests.routes.test_auth_sms_02_portal_contract import _make_portal_user


def _open_session(db, user: User, *, required: bool) -> ActivationSession:
    user.account_status = "pending_activation"
    for client in user.clients:
        client.is_active = False
    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = user.id
    session.email_verified_at = datetime.now(UTC)
    session.portal_terms_required = required
    db.session.add(session)
    db.session.commit()
    return session


def _mark_phone_verified(db, user: User, session: ActivationSession) -> None:
    now = datetime.now(UTC)
    user.phone_verified_at = now
    session.phone_verified_at = now
    db.session.commit()


def test_new_portal_acceptance_creates_two_rows_matching_catalog(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=True)
    _mark_phone_verified(db, user, session)

    listed = client.get(
        "/api/v1/auth/activation/portal-terms",
        query_string={"activation_session_id": session.activation_session_id},
    )
    assert listed.status_code == 200, listed.get_json()
    documents = listed.get_json()["documents"]

    refused = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert refused.status_code == 400
    assert refused.get_json()["error"] == "terms_acceptance_required"
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0

    done = client.post(
        "/api/v1/auth/activation/finalize",
        json={
            "activation_session_id": session.activation_session_id,
            "accept_current_portal_terms": True,
        },
    )
    assert done.status_code == 200, done.get_json()
    rows = ClientTermsAcceptance.query.filter_by(user_id=user.id).all()
    assert len(rows) == 2
    assert {row.document_type for row in rows} == {
        "terms_of_service",
        "transport_terms",
    }
    assert {row.terms_version for row in rows} == {"1.0"}
    assert {(doc["document_type"], doc["terms_hash"]) for doc in documents} == {
        (row.document_type, row.terms_hash) for row in rows
    }
    catalog = {
        (spec.document_type, spec.terms_hash) for spec in current_portal_terms()
    }
    assert {(row.document_type, row.terms_hash) for row in rows} == catalog
    db.session.refresh(user)
    assert user.account_status == "active"
    assert user.phone_verified_at is not None
    assert all(row.verification_method == VERIFICATION_OTP_SMS for row in rows)
    assert all(row.phone_verified_at_snapshot is not None for row in rows)
    assert all(row.phone_snapshot == user.phone for row in rows)


def test_client_cannot_choose_version_or_hash(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=True)
    response = client.post(
        "/api/v1/auth/activation/finalize",
        json={
            "activation_session_id": session.activation_session_id,
            "accept_current_portal_terms": True,
            "terms_version": "9.9",
            "terms_hash": "a" * 64,
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "client_supplied_terms_forbidden"
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0
    db.session.refresh(user)
    assert user.account_status == "pending_activation"


def test_second_insert_failure_rolls_back_finalization(client, db, monkeypatch):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=True)
    _mark_phone_verified(db, user, session)

    def fail_second(portal_user, portal_client, documents=None):
        specs = list(documents) if documents is not None else list(current_portal_terms())
        record_portal_terms_acceptance(portal_user, portal_client, documents=specs[:1])
        raise RuntimeError("insertion transport_terms interrompue")

    monkeypatch.setattr(
        "services.legal.record_terms_acceptance.record_portal_terms_acceptance",
        fail_second,
    )
    response = client.post(
        "/api/v1/auth/activation/finalize",
        json={
            "activation_session_id": session.activation_session_id,
            "accept_current_portal_terms": True,
        },
    )
    assert response.status_code == 500
    db.session.expire_all()
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0
    fresh_user = db.session.get(User, user.id)
    fresh_session = db.session.get(ActivationSession, session.id)
    assert fresh_user is not None
    assert fresh_user.account_status == "pending_activation"
    assert fresh_session is not None
    assert fresh_session.consumed_at is None


def test_repeat_finalize_does_not_duplicate_rows(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=True)
    _mark_phone_verified(db, user, session)
    payload = {
        "activation_session_id": session.activation_session_id,
        "accept_current_portal_terms": True,
    }
    first = client.post("/api/v1/auth/activation/finalize", json=payload)
    second = client.post("/api/v1/auth/activation/finalize", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 2
    assert "with_for_update" in inspect.getsource(auth.FinalizeActivation)


def test_verified_phone_is_recorded_as_otp_sms(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=True)
    _mark_phone_verified(db, user, session)
    response = client.post(
        "/api/v1/auth/activation/finalize",
        json={
            "activation_session_id": session.activation_session_id,
            "accept_current_portal_terms": True,
        },
    )
    assert response.status_code == 200, response.get_json()
    rows = ClientTermsAcceptance.query.filter_by(user_id=user.id).all()
    assert len(rows) == 2
    assert all(row.verification_method == VERIFICATION_OTP_SMS for row in rows)
    assert all(row.phone_verified_at_snapshot is not None for row in rows)


def test_later_phone_verification_does_not_rewrite_acceptance(client, db):
    user, portal = _make_portal_user(db, pending=True)
    record_portal_terms_acceptance(user, portal)
    db.session.commit()
    sms_source = inspect.getsource(auth.VerifyActivationSms)
    assert "ClientTermsAcceptance" not in sms_source
    user.phone_verified_at = datetime.now(UTC)
    db.session.commit()
    rows = ClientTermsAcceptance.query.filter_by(user_id=user.id).all()
    assert len(rows) == 2
    assert all(row.verification_method == VERIFICATION_NOT_VERIFIED for row in rows)
    assert all(row.phone_verified_at_snapshot is None for row in rows)


def test_existing_session_is_not_backfilled(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=False)
    response = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert response.status_code == 200, response.get_json()
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0
    db.session.refresh(user)
    assert user.account_status == "active"
    assert user.phone_verified_at is None


def test_new_account_finalize_without_phone_is_refused(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    session = _open_session(db, user, required=True)
    response = client.post(
        "/api/v1/auth/activation/finalize",
        json={
            "activation_session_id": session.activation_session_id,
            "accept_current_portal_terms": True,
        },
    )
    assert response.status_code == 400, response.get_json()
    db.session.refresh(user)
    assert user.account_status == "pending_activation"
    assert user.phone_verified_at is None
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0


def test_login_does_not_skip_terms_for_a_new_account(client, db):
    user, _portal = _make_portal_user(db, pending=True)
    _open_session(db, user, required=True)
    response = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "Password123!"},
    )
    assert response.status_code == 403
    db.session.refresh(user)
    assert user.account_status == "pending_activation"
    assert ClientTermsAcceptance.query.filter_by(user_id=user.id).count() == 0

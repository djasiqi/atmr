"""Tests POST /api/v1/auth/session-resume (rotation credential + idempotence)."""

from __future__ import annotations

import uuid

import pytest

from ext import db
from models import User
from models.enums import UserRole
from security.mobile_device_session_service import create_or_reuse_session

SESSION_RESUME_URL = "/api/v1/auth/session-resume"


@pytest.fixture
def resume_user(db):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"resume_{suffix}",
        email=f"resume_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.driver,
    )
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def resume_session(client, resume_user):
    device_installation_id = f"device-{uuid.uuid4()}"
    with client.application.app_context():
        session, recovery, revocation = create_or_reuse_session(
            user_id=resume_user.id,
            device_installation_id=device_installation_id,
            role="driver",
        )
        db.session.commit()
        session_id = str(session.session_id)
    return {
        "session_id": session_id,
        "device_installation_id": device_installation_id,
        "recovery_credential": recovery,
        "revocation_secret": revocation,
    }


def test_session_resume_happy_path(client, db, resume_user, resume_session):
    response = client.post(
        SESSION_RESUME_URL,
        json={
            "session_id": resume_session["session_id"],
            "device_installation_id": resume_session["device_installation_id"],
            "recovery_credential": resume_session["recovery_credential"],
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["session_id"] == resume_session["session_id"]
    assert body["access_token"]
    assert body["refresh_token"]
    # Rotation : le recovery_credential renvoyé diffère du précédent.
    assert body["recovery_credential"] != resume_session["recovery_credential"]
    assert body["auth_contract_version"] == "mobile-device-session-v1"
    assert body["capabilities"]["session_resume"] is True


def test_session_resume_idempotent_replay(client, db, resume_user, resume_session):
    idempotency_key = str(uuid.uuid4())

    first = client.post(
        SESSION_RESUME_URL,
        json={
            "session_id": resume_session["session_id"],
            "device_installation_id": resume_session["device_installation_id"],
            "recovery_credential": resume_session["recovery_credential"],
            "idempotency_key": idempotency_key,
        },
    )
    assert first.status_code == 200
    first_body = first.get_json()

    second = client.post(
        SESSION_RESUME_URL,
        json={
            "session_id": resume_session["session_id"],
            "device_installation_id": resume_session["device_installation_id"],
            "recovery_credential": resume_session["recovery_credential"],
            "idempotency_key": idempotency_key,
        },
    )
    assert second.status_code == 200
    second_body = second.get_json()

    # Rejoue la même réponse chiffrée : mêmes tokens, error_code de duplication.
    assert second_body["error_code"] == "refresh_duplicate"
    assert second_body["access_token"] == first_body["access_token"]
    assert second_body["refresh_token"] == first_body["refresh_token"]
    assert second_body["recovery_credential"] == first_body["recovery_credential"]


def test_session_resume_invalid_credential_rejected(client, db, resume_session):
    response = client.post(
        SESSION_RESUME_URL,
        json={
            "session_id": resume_session["session_id"],
            "device_installation_id": resume_session["device_installation_id"],
            "recovery_credential": "credential-invalide",
        },
    )

    assert response.status_code == 401
    body = response.get_json()
    assert body["error_code"] == "session_revoked"


def test_session_resume_missing_params_returns_400(client, db):
    response = client.post(SESSION_RESUME_URL, json={"session_id": str(uuid.uuid4())})

    assert response.status_code == 400
    body = response.get_json()
    assert body["error_code"] == "invalid_request"


def test_session_resume_installation_mismatch_rejected(client, db, resume_session):
    response = client.post(
        SESSION_RESUME_URL,
        json={
            "session_id": resume_session["session_id"],
            "device_installation_id": f"other-device-{uuid.uuid4()}",
            "recovery_credential": resume_session["recovery_credential"],
        },
    )

    assert response.status_code == 401
    body = response.get_json()
    assert body["error_code"] == "refresh_replay_detected"

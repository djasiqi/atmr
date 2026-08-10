"""Tests POST /api/v1/auth/session-resume (rotation credential + idempotence)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from ext import db
from models import User
from models.enums import UserRole
from models.mobile_device_session import AuthRotationResult
from security import mobile_device_session_service as svc
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
        session, recovery, revocation, _ = create_or_reuse_session(
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


def test_session_resume_replay_after_grace_expired(
    client, db, resume_session, monkeypatch
):
    """Receipt valide + grâce credential expirée → replay sans verify courant."""
    monkeypatch.setenv("MOBILE_SESSION_PREVIOUS_CREDENTIAL_GRACE_SECONDS", "0")
    # Recharger la constante déjà lue au import
    monkeypatch.setattr(svc, "PREVIOUS_CREDENTIAL_GRACE_SECONDS", 0)

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

    # Forcer expiration de la grâce previous_credential
    with client.application.app_context():
        from security.mobile_device_session_service import get_session_by_id

        session = get_session_by_id(resume_session["session_id"])
        assert session is not None
        session.previous_credential_valid_until = datetime.now(UTC) - timedelta(
            seconds=1
        )
        db.session.commit()

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
    assert second_body["error_code"] == "refresh_duplicate"
    assert second_body["access_token"] == first_body["access_token"]
    assert second_body["recovery_credential"] == first_body["recovery_credential"]


def test_session_resume_replay_with_successor_credential(client, db, resume_session):
    """Après crash local ayant écrit le nouveau recovery : même clé → replay."""
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
            "recovery_credential": first_body["recovery_credential"],
            "idempotency_key": idempotency_key,
        },
    )
    assert second.status_code == 200
    assert second.get_json()["error_code"] == "refresh_duplicate"


def test_session_resume_expired_receipt_no_mutation(client, db, resume_session):
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
    new_recovery = first.get_json()["recovery_credential"]

    with client.application.app_context():
        row = AuthRotationResult.query.filter_by(
            session_id=uuid.UUID(resume_session["session_id"]),
            idempotency_key_hash=svc.hash_idempotency_key(idempotency_key),
        ).one()
        row.expires_at = datetime.now(UTC) - timedelta(seconds=5)
        gen_before = svc.get_session_by_id(resume_session["session_id"]).generation
        db.session.commit()

    expired = client.post(
        SESSION_RESUME_URL,
        json={
            "session_id": resume_session["session_id"],
            "device_installation_id": resume_session["device_installation_id"],
            "recovery_credential": new_recovery,
            "idempotency_key": idempotency_key,
        },
    )
    assert expired.status_code == 401
    assert expired.get_json()["error_code"] == "idempotency_result_expired"

    with client.application.app_context():
        session = svc.get_session_by_id(resume_session["session_id"])
        assert session.generation == gen_before
        assert (
            AuthRotationResult.query.filter_by(
                session_id=uuid.UUID(resume_session["session_id"]),
                idempotency_key_hash=svc.hash_idempotency_key(idempotency_key),
            ).count()
            == 1
        )


def test_session_resume_unreadable_receipt_returns_503(client, db, resume_session):
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

    with patch.object(
        svc, "decrypt_rotation_response", side_effect=RuntimeError("bad key")
    ):
        second = client.post(
            SESSION_RESUME_URL,
            json={
                "session_id": resume_session["session_id"],
                "device_installation_id": resume_session["device_installation_id"],
                "recovery_credential": resume_session["recovery_credential"],
                "idempotency_key": idempotency_key,
            },
        )
    assert second.status_code == 503
    body = second.get_json()
    assert body["error_code"] == "rotation_result_unavailable"
    assert body["retryable"] is True


def test_session_resume_ttl_longer_than_refresh(monkeypatch):
    monkeypatch.setenv("AUTH_ROTATION_RESULT_TTL_SECONDS", "600")
    monkeypatch.setenv("AUTH_SESSION_RESUME_RESULT_TTL_SECONDS", "86400")
    monkeypatch.setattr(svc, "ROTATION_RESULT_TTL_SECONDS", 600)
    monkeypatch.setattr(svc, "SESSION_RESUME_RESULT_TTL_SECONDS", 86400)
    assert svc.rotation_result_ttl_seconds("refresh") == 600
    assert svc.rotation_result_ttl_seconds("session_resume") == 86400


def test_is_rotation_idempotency_conflict_requires_constraint_name():
    class _Diag:
        constraint_name = "uq_auth_rotation_result_session_idempotency"

    class _Orig(Exception):
        pgcode = "23505"
        diag = _Diag()

    from sqlalchemy.exc import IntegrityError

    exc = IntegrityError("stmt", {}, _Orig())
    assert svc.is_rotation_idempotency_conflict(exc) is True

    class _OtherDiag:
        constraint_name = "some_other_unique"

    class _OtherOrig(Exception):
        pgcode = "23505"
        diag = _OtherDiag()

    other = IntegrityError("stmt", {}, _OtherOrig())
    assert svc.is_rotation_idempotency_conflict(other) is False


@pytest.mark.integration
def test_session_resume_concurrent_same_idempotency_key(app, db, resume_session):
    """Deux POST simultanés (même session + Idempotency-Key) → 1 rotation, 0 × 500."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Barrier

    session_id = resume_session["session_id"]
    payload = {
        "session_id": session_id,
        "device_installation_id": resume_session["device_installation_id"],
        "recovery_credential": resume_session["recovery_credential"],
        "idempotency_key": str(uuid.uuid4()),
    }

    with app.app_context():
        session_before = svc.get_session_by_id(session_id)
        assert session_before is not None
        initial_credential_generation = int(session_before.credential_generation or 1)
        initial_generation = int(session_before.generation or 1)
        initial_refresh_generation = int(session_before.refresh_generation or 1)
        initial_session_epoch = int(session_before.session_epoch or 1)

    barrier = Barrier(2)

    def _post_once() -> tuple[int, dict]:
        barrier.wait(timeout=30)
        with app.test_client() as thread_client:
            response = thread_client.post(SESSION_RESUME_URL, json=payload)
            body = response.get_json() or {}
            return response.status_code, body

    results: list[tuple[int, dict]] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_post_once) for _ in range(2)]
        for future in as_completed(futures):
            results.append(future.result())

    assert len(results) == 2
    assert all(status == 200 for status, _ in results), results
    assert not any(status >= 500 for status, _ in results)

    bodies = [body for _, body in results]
    tokens = {(b.get("access_token"), b.get("refresh_token"), b.get("recovery_credential")) for b in bodies}
    assert len(tokens) == 1, "les deux réponses doivent rejouer le même payload gagnant"
    assert any(b.get("error_code") == "refresh_duplicate" for b in bodies) or all(
        b.get("access_token") for b in bodies
    )

    # Relecture PostgreSQL hors identity map (objets pré-course invalidés).
    db.session.remove()
    with app.app_context():
        session_after = svc.get_session_by_id(session_id)
        assert session_after is not None
        assert session_after.credential_generation == initial_credential_generation + 1
        assert session_after.generation == initial_generation + 1
        assert session_after.refresh_generation == initial_refresh_generation
        assert session_after.session_epoch == initial_session_epoch

        key_hash = svc.hash_idempotency_key(payload["idempotency_key"])
        rows = AuthRotationResult.query.filter_by(
            session_id=uuid.UUID(session_id),
            idempotency_key_hash=key_hash,
        ).all()
        assert len(rows) == 1

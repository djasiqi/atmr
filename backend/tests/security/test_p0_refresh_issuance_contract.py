"""P0-1 / P0-3 — contrat d'émission refresh (DB + Redis + génération resume)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from flask_jwt_extended import decode_token

from ext import db
from models import RefreshToken, User
from models.enums import UserRole
from security import mobile_device_session_service as svc
from security.mobile_device_session_service import create_or_reuse_session
from security.refresh_token_service import (
    RefreshStoreUnavailableError,
    refresh_fail_closed_enabled,
    store_refresh_token,
    sync_refresh_token_to_redis,
)
from services.security.authentication import RefreshTokenService

SESSION_RESUME_URL = "/api/v1/auth/session-resume"
REFRESH_URL = "/api/v1/auth/refresh-token"


@pytest.fixture
def fail_closed(monkeypatch):
    monkeypatch.setenv("REFRESH_FAIL_CLOSED", "true")


@pytest.fixture
def resume_user(db):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"p0_resume_{suffix}",
        email=f"p0_resume_{suffix}@test.local",
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
        refresh_gen = int(session.refresh_generation or 1)
    return {
        "session_id": session_id,
        "device_installation_id": device_installation_id,
        "recovery_credential": recovery,
        "revocation_secret": revocation,
        "refresh_generation": refresh_gen,
    }


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _redis_has_refresh(token: str) -> bool:
    svc_rt = RefreshTokenService()
    key = f"{svc_rt.active_tokens_prefix}{_token_hash(token)}"
    return bool(svc_rt.redis_client.get(key))


class TestP0_1StoreHelper:
    def test_sync_redis_fail_closed_raises(self, app, fail_closed):
        with app.app_context():
            assert refresh_fail_closed_enabled() is True
            with patch(
                "services.security.authentication.RefreshTokenService.store_token",
                side_effect=RuntimeError("redis down"),
            ):
                with pytest.raises(RefreshStoreUnavailableError):
                    sync_refresh_token_to_redis(1, "fake-jwt-token", ttl_seconds=60)

    def test_store_commit_true_syncs_redis(self, app, db, resume_user, fail_closed):
        with app.app_context():
            token = f"unit-refresh-{uuid.uuid4()}"
            # JWT non requis pour le hash Redis / ligne DB
            expires = datetime.now(UTC) + timedelta(days=1)
            with patch(
                "services.security.authentication.RefreshTokenService.store_token"
            ) as store_mock:
                store_refresh_token(
                    token=token,
                    user_id=resume_user.id,
                    expires_at=expires,
                    device_id="dev-1",
                    commit=True,
                )
                store_mock.assert_called_once()
            row = RefreshToken.query.filter_by(
                token_hash=_token_hash(token), is_revoked=False
            ).first()
            assert row is not None

    def test_store_commit_true_redis_fail_revokes_db(
        self, app, db, resume_user, fail_closed
    ):
        with app.app_context():
            token = f"unit-refresh-fail-{uuid.uuid4()}"
            expires = datetime.now(UTC) + timedelta(days=1)
            with patch(
                "services.security.authentication.RefreshTokenService.store_token",
                side_effect=RuntimeError("redis down"),
            ):
                with pytest.raises(RefreshStoreUnavailableError):
                    store_refresh_token(
                        token=token,
                        user_id=resume_user.id,
                        expires_at=expires,
                        commit=True,
                    )
            row = RefreshToken.query.filter_by(token_hash=_token_hash(token)).first()
            assert row is not None
            assert row.is_revoked is True
            assert row.revoked_reason == "redis_sync_failed"


class TestP0_1P0_3SessionResume:
    def test_a_resume_then_refresh_accepted(
        self, client, db, resume_user, resume_session, fail_closed
    ):
        """A: resume → refresh DB+Redis → refresh-token suivant accepté."""
        resume = client.post(
            SESSION_RESUME_URL,
            json={
                "session_id": resume_session["session_id"],
                "device_installation_id": resume_session["device_installation_id"],
                "recovery_credential": resume_session["recovery_credential"],
            },
        )
        assert resume.status_code == 200, resume.get_json()
        body = resume.get_json()
        refresh = body["refresh_token"]
        assert body["refresh_generation"] == resume_session["refresh_generation"] + 1

        row = RefreshToken.query.filter_by(
            token_hash=_token_hash(refresh), is_revoked=False
        ).first()
        assert row is not None
        assert _redis_has_refresh(refresh) is True

        claims = decode_token(refresh)
        assert claims.get("refresh_generation") == body["refresh_generation"]

        nxt = client.post(
            REFRESH_URL,
            json={"refresh_token": refresh},
            headers={
                "X-Requested-With": "Expo",
                "X-Device-ID": resume_session["device_installation_id"],
                "X-Client-Platform": "ios",
            },
        )
        assert nxt.status_code == 200, nxt.get_json()
        nxt_body = nxt.get_json()
        assert nxt_body.get("refresh_token") or nxt_body.get("access_token")

    def test_b_generation_bumped(
        self, client, db, resume_user, resume_session, fail_closed
    ):
        """B: generation N → resume → N+1 aligné serveur + JWT."""
        before = resume_session["refresh_generation"]
        resume = client.post(
            SESSION_RESUME_URL,
            json={
                "session_id": resume_session["session_id"],
                "device_installation_id": resume_session["device_installation_id"],
                "recovery_credential": resume_session["recovery_credential"],
            },
        )
        assert resume.status_code == 200
        body = resume.get_json()
        assert body["refresh_generation"] == before + 1

        db.session.remove()
        session = svc.get_session_by_id(resume_session["session_id"])
        assert session is not None
        assert int(session.refresh_generation) == before + 1
        assert decode_token(body["refresh_token"]).get("refresh_generation") == before + 1

    def test_d_redis_unavailable_no_token_to_client(
        self, client, db, resume_user, resume_session, fail_closed
    ):
        """D: fail-closed — aucun refresh remis si Redis down."""
        with patch(
            "routes.auth_mobile_session.publish_refresh_redis",
            side_effect=RefreshStoreUnavailableError("redis_unavailable"),
        ):
            resume = client.post(
                SESSION_RESUME_URL,
                json={
                    "session_id": resume_session["session_id"],
                    "device_installation_id": resume_session[
                        "device_installation_id"
                    ],
                    "recovery_credential": resume_session["recovery_credential"],
                },
            )
        assert resume.status_code == 503
        body = resume.get_json() or {}
        assert body.get("error_code") == "store_unavailable"
        assert "refresh_token" not in body

        db.session.remove()
        session = svc.get_session_by_id(resume_session["session_id"])
        assert session is not None
        # Rollback : génération non bumpée côté vérité.
        assert int(session.refresh_generation or 1) == resume_session[
            "refresh_generation"
        ]

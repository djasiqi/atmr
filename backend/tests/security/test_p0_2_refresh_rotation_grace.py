"""P0-1 hardening + P0-2 — rotation Redis grâce / idempotence / compensation."""

from __future__ import annotations

import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from threading import Barrier
from unittest.mock import patch

import pytest
from flask_jwt_extended import create_refresh_token, decode_token

from models import RefreshToken, User
from models.enums import UserRole
from security.mobile_device_session_service import create_or_reuse_session
from security.refresh_redis_rotation import (
    RedisRefreshState,
    classify_refresh_in_redis,
    commit_db_after_redis,
    compensate_redis_issuance,
    publish_refresh_redis,
    rotate_refresh_redis,
    rotation_grace_seconds,
)
from security.refresh_token_service import (
    RefreshStoreUnavailableError,
    is_token_revoked,
    mark_token_rotated,
    store_refresh_token,
    update_token_last_used,
)

REFRESH_URL = "/api/v1/auth/refresh-token"


@pytest.fixture
def fail_closed(monkeypatch):
    monkeypatch.setenv("REFRESH_FAIL_CLOSED", "true")


@pytest.fixture
def driver_user(db):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"p02_{suffix}",
        email=f"p02_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.driver,
    )
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.commit()
    return user


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _mobile_headers(device_id: str, idem: str | None = None) -> dict:
    h = {
        "X-Requested-With": "Expo",
        "X-Client-Platform": "ios",
        "X-Device-ID": device_id,
        "X-Auth-Contract-Version": "mobile-device-session-v1",
    }
    if idem:
        h["Idempotency-Key"] = idem
    return h


def _login_mobile(client, user, device_id: str) -> dict:
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "password123"},
        headers=_mobile_headers(device_id),
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body.get("refresh_token")
    assert body.get("session_id")
    return body


class TestP0_1CommitCompensation:
    def test_g_redis_ok_commit_fail_restores_previous(
        self, app, db, driver_user, fail_closed
    ):
        """G: Redis rotate OK → commit FAIL → ancien CURRENT, nouveau absent."""
        with app.app_context():
            device_id = f"dev-{uuid.uuid4()}"
            session, _rec, _rev, _ = create_or_reuse_session(
                user_id=driver_user.id,
                device_installation_id=device_id,
                role="driver",
            )
            db.session.commit()
            expires = datetime.now(UTC) + timedelta(days=30)
            r0 = f"r0-{uuid.uuid4()}"
            r1 = f"r1-{uuid.uuid4()}"
            store_refresh_token(
                r0,
                driver_user.id,
                expires,
                device_id=device_id,
                sync_redis=True,
            )
            assert (
                classify_refresh_in_redis(r0, user_id=driver_user.id).state
                == RedisRefreshState.CURRENT
            )

            handle = rotate_refresh_redis(
                driver_user.id,
                r0,
                r1,
                ttl_seconds=3600,
                session_id=str(session.session_id),
            )
            assert (
                classify_refresh_in_redis(r1, user_id=driver_user.id).state
                == RedisRefreshState.CURRENT
            )
            assert (
                classify_refresh_in_redis(r0, user_id=driver_user.id).state
                == RedisRefreshState.PREVIOUS_WITHIN_GRACE
            )

            with (
                patch.object(db.session, "commit", side_effect=RuntimeError("boom")),
                pytest.raises(RuntimeError),
            ):
                commit_db_after_redis(handle)

            assert (
                classify_refresh_in_redis(r0, user_id=driver_user.id).state
                == RedisRefreshState.CURRENT
            )
            assert (
                classify_refresh_in_redis(r1, user_id=driver_user.id).state
                == RedisRefreshState.UNKNOWN
            )

    def test_publish_commit_fail_removes_new(self, app, db, driver_user, fail_closed):
        with app.app_context():
            token = f"pub-{uuid.uuid4()}"
            handle = publish_refresh_redis(driver_user.id, token, ttl_seconds=600)
            assert (
                classify_refresh_in_redis(token, user_id=driver_user.id).state
                == RedisRefreshState.CURRENT
            )
            with (
                patch.object(db.session, "commit", side_effect=RuntimeError("fail")),
                pytest.raises(RuntimeError),
            ):
                commit_db_after_redis(handle)
            assert (
                classify_refresh_in_redis(token, user_id=driver_user.id).state
                == RedisRefreshState.UNKNOWN
            )


class TestP0_2RedisGrace:
    def test_classify_previous_within_and_expired(
        self, app, db, driver_user, fail_closed, monkeypatch
    ):
        monkeypatch.setenv("REFRESH_ROTATION_GRACE_SECONDS", "2")
        with app.app_context():
            assert rotation_grace_seconds() == 2
            r0, r1 = f"old-{uuid.uuid4()}", f"new-{uuid.uuid4()}"
            publish_refresh_redis(driver_user.id, r0, ttl_seconds=600)
            rotate_refresh_redis(
                driver_user.id, r0, r1, ttl_seconds=600, grace_seconds=2
            )
            assert (
                classify_refresh_in_redis(r0, user_id=driver_user.id).state
                == RedisRefreshState.PREVIOUS_WITHIN_GRACE
            )
            time.sleep(2.2)
            # TTL Redis a expiré la clé previous
            state = classify_refresh_in_redis(r0, user_id=driver_user.id).state
            assert state in (
                RedisRefreshState.UNKNOWN,
                RedisRefreshState.EXPIRED_PREVIOUS,
            )


class TestP0_2SameDeviceBypass:
    def test_superseded_same_device_outside_grace_rejected(
        self, app, db, driver_user, monkeypatch
    ):
        with app.app_context():
            expires = datetime.now(UTC) + timedelta(days=30)
            old_t, new_t = f"old-{uuid.uuid4()}", f"new-{uuid.uuid4()}"
            store_refresh_token(
                old_t, driver_user.id, expires, device_id="dev-a", sync_redis=False
            )
            store_refresh_token(
                new_t, driver_user.id, expires, device_id="dev-a", sync_redis=False
            )
            mark_token_rotated(old_t, new_t)
            update_token_last_used(new_t)
            row = RefreshToken.query.filter_by(token_hash=_sha(old_t)).first()
            assert row is not None
            from security.refresh_token_service import _supersede_old_token

            _supersede_old_token(row)
            row = RefreshToken.query.filter_by(token_hash=_sha(old_t)).first()
            assert row is not None
            row.revoked_at = datetime.now(UTC) - timedelta(seconds=400)
            row.rotated_at = datetime.now(UTC) - timedelta(seconds=400)
            db.session.commit()
            assert (
                is_token_revoked(old_t, request_device_id="dev-a", grace_window=True)
                is True
            )

    def test_reuse_same_device_within_grace_no_revoke_all(
        self, app, db, driver_user, monkeypatch
    ):
        with app.app_context():
            expires = datetime.now(UTC) + timedelta(days=30)
            old_t, new_t = f"old-{uuid.uuid4()}", f"new-{uuid.uuid4()}"
            store_refresh_token(
                old_t, driver_user.id, expires, device_id="dev-a", sync_redis=False
            )
            store_refresh_token(
                new_t, driver_user.id, expires, device_id="dev-a", sync_redis=False
            )
            mark_token_rotated(old_t, new_t)
            update_token_last_used(new_t)
            called = {"n": 0}

            def _fake_revoke_all(user_id, reason=None, **_k):
                called["n"] += 1
                return 0

            monkeypatch.setattr(
                "security.refresh_token_service.revoke_all_user_tokens",
                _fake_revoke_all,
            )
            assert (
                is_token_revoked(old_t, request_device_id="dev-a", grace_window=True)
                is False
            )
            assert called["n"] == 0


@pytest.mark.integration
class TestP0_2RefreshEndpoint:
    def test_a_b_lost_response_returns_same_r1(
        self, client, db, driver_user, fail_closed
    ):
        device_id = f"dev-{uuid.uuid4()}"
        login = _login_mobile(client, driver_user, device_id)
        r0 = login["refresh_token"]
        idem = str(uuid.uuid4())
        first = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_id, idem),
        )
        assert first.status_code == 200, first.get_json()
        body1 = first.get_json()
        r1 = body1["refresh_token"]
        assert r1 != r0
        gen1 = body1.get("refresh_generation")

        # Retry exact (réponse « perdue »)
        second = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_id, idem),
        )
        assert second.status_code == 200, second.get_json()
        body2 = second.get_json()
        assert body2["refresh_token"] == r1
        assert body2.get("refresh_generation") == gen1
        assert body2.get("error_code") == "refresh_duplicate"

    def test_c_previous_within_grace_no_fork(
        self, client, db, driver_user, fail_closed
    ):
        device_id = f"dev-{uuid.uuid4()}"
        login = _login_mobile(client, driver_user, device_id)
        r0 = login["refresh_token"]
        idem = str(uuid.uuid4())
        first = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_id, idem),
        )
        assert first.status_code == 200
        r1 = first.get_json()["refresh_token"]
        gen = first.get_json().get("refresh_generation")

        # Même R0, autre Idempotency-Key → recovery via previous, pas R2
        other_idem = str(uuid.uuid4())
        retry = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_id, other_idem),
        )
        assert retry.status_code == 200, retry.get_json()
        body = retry.get_json()
        assert body["refresh_token"] == r1
        assert body.get("refresh_generation") == gen

    def test_e_different_idem_key_no_r2(self, client, db, driver_user, fail_closed):
        self.test_c_previous_within_grace_no_fork(client, db, driver_user, fail_closed)

    def test_d_previous_outside_grace_rejected(
        self, client, db, driver_user, fail_closed, monkeypatch
    ):
        monkeypatch.setenv("REFRESH_ROTATION_GRACE_SECONDS", "1")
        device_id = f"dev-{uuid.uuid4()}"
        login = _login_mobile(client, driver_user, device_id)
        r0 = login["refresh_token"]
        first = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_id, str(uuid.uuid4())),
        )
        assert first.status_code == 200
        time.sleep(1.3)
        retry = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_id, str(uuid.uuid4())),
        )
        assert retry.status_code == 401

    def test_f_concurrent_same_r0(self, app, client, db, driver_user, fail_closed):
        device_id = f"dev-{uuid.uuid4()}"
        login = _login_mobile(client, driver_user, device_id)
        r0 = login["refresh_token"]
        idem = str(uuid.uuid4())
        barrier = Barrier(2)

        def _once():
            barrier.wait(timeout=30)
            with app.test_client() as c:
                return c.post(
                    REFRESH_URL,
                    json={"refresh_token": r0},
                    headers=_mobile_headers(device_id, idem),
                )

        results = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_once) for _ in range(2)]
            for f in as_completed(futures):
                results.append(f.result())

        assert all(r.status_code == 200 for r in results), [
            (r.status_code, r.get_json()) for r in results
        ]
        tokens = {r.get_json().get("refresh_token") for r in results}
        assert len(tokens) == 1

    def test_h_cross_device_rejected(self, client, db, driver_user, fail_closed):
        device_a = f"dev-a-{uuid.uuid4()}"
        device_b = f"dev-b-{uuid.uuid4()}"
        login = _login_mobile(client, driver_user, device_a)
        r0 = login["refresh_token"]
        resp = client.post(
            REFRESH_URL,
            json={"refresh_token": r0},
            headers=_mobile_headers(device_b, str(uuid.uuid4())),
        )
        assert resp.status_code == 401
        assert resp.get_json().get("error_code") in (
            "refresh_replay_detected",
            "session_expired",
            "installation_mismatch",
        )

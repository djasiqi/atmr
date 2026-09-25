"""P0-6 — Redis auth dédié + anti-éviction + OOM fail-closed + anti-régression FOR UPDATE."""

from __future__ import annotations

import hashlib
import os
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import redis
from redis.connection import ConnectionPool

from security.auth_redis import (
    auth_redis_write,
    get_auth_redis,
    is_redis_oom_error,
    reset_auth_redis_client,
    resolve_auth_redis_url,
)
from security.refresh_token_service import RefreshStoreUnavailableError, update_token_last_used


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _real_redis_from_url(url: str, **kwargs):
    """Construit un client Redis réel (contourne mock autouse conftest)."""
    decode = kwargs.pop("decode_responses", True)
    pool = ConnectionPool.from_url(url, decode_responses=decode, **kwargs)
    return redis.Redis(connection_pool=pool)


class TestAuthRedisResolver:
    def test_auth_redis_url_preferred(self, app):
        reset_auth_redis_client()
        with app.app_context():
            app.config["AUTH_REDIS_URL"] = "redis://auth-dedicated:6379/0"
            app.config["REDIS_URL"] = "redis://general:6379/0"
            assert resolve_auth_redis_url() == "redis://auth-dedicated:6379/0"

    def test_fallback_redis_url(self, app, monkeypatch):
        reset_auth_redis_client()
        monkeypatch.delenv("AUTH_REDIS_ALLOW_LEGACY_FALLBACK", raising=False)
        monkeypatch.delenv("AUTH_REDIS_URL", raising=False)
        monkeypatch.setenv("FLASK_ENV", "testing")
        with app.app_context():
            app.config["AUTH_REDIS_URL"] = ""
            app.config["REDIS_URL"] = "redis://general:6379/0"
            app.config["ENV"] = "testing"
            app.config["TESTING"] = True
            assert resolve_auth_redis_url() == "redis://general:6379/0"

    def test_production_missing_auth_redis_url_raises(self, app, monkeypatch):
        reset_auth_redis_client()
        monkeypatch.delenv("AUTH_REDIS_ALLOW_LEGACY_FALLBACK", raising=False)
        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.delenv("AUTH_REDIS_URL", raising=False)
        with app.app_context():
            app.config["AUTH_REDIS_URL"] = ""
            app.config["REDIS_URL"] = "redis://general:6379/0"
            app.config["ENV"] = "production"
            app.config["TESTING"] = False
            app.config["DEBUG"] = False
            with pytest.raises(RuntimeError, match="AUTH_REDIS_URL"):
                resolve_auth_redis_url()

    def test_production_legacy_fallback_flag_allows(self, app, monkeypatch):
        reset_auth_redis_client()
        monkeypatch.setenv("AUTH_REDIS_ALLOW_LEGACY_FALLBACK", "1")
        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.delenv("AUTH_REDIS_URL", raising=False)
        with app.app_context():
            app.config["AUTH_REDIS_URL"] = ""
            app.config["REDIS_URL"] = "redis://general:6379/0"
            app.config["ENV"] = "production"
            app.config["TESTING"] = False
            assert resolve_auth_redis_url() == "redis://general:6379/0"


class TestOomFailClosed:
    def test_is_redis_oom_error(self):
        assert is_redis_oom_error(Exception("OOM command not allowed"))
        assert is_redis_oom_error(
            redis.ResponseError("OOM command not allowed when used memory")
        )
        assert not is_redis_oom_error(Exception("WRONGTYPE"))

    def test_auth_redis_write_oom_raises_store_unavailable(self, app, fail_closed_env):
        with app.app_context():

            def _boom():
                raise redis.ResponseError(
                    "OOM command not allowed when used memory > 'maxmemory'"
                )

            with pytest.raises(RefreshStoreUnavailableError) as exc:
                auth_redis_write(_boom)
            assert "oom" in str(exc.value).lower() or "redis" in str(exc.value).lower()


@pytest.fixture
def fail_closed_env(monkeypatch):
    monkeypatch.setenv("REFRESH_FAIL_CLOSED", "true")


class TestForUpdateNoImplicitCommit:
    """Invariant : sous FOR UPDATE rotation, pas de commit() implicite via helpers."""

    def test_update_token_last_used_commit_false_does_not_commit(self, app, db):
        from flask_jwt_extended import create_refresh_token

        from models import RefreshToken, User
        from models.enums import UserRole
        from security.refresh_token_service import store_refresh_token

        with app.app_context():
            user = User(
                username=f"fu_{uuid.uuid4().hex[:8]}",
                email=f"fu_{uuid.uuid4().hex[:8]}@t.local",
                public_id=str(uuid.uuid4()),
                role=UserRole.driver,
            )
            user.set_password("password123", force_change=False)
            db.session.add(user)
            db.session.commit()

            tok = create_refresh_token(
                identity=str(user.public_id),
                additional_claims={"aud": "atmr-api"},
            )
            store_refresh_token(
                tok,
                user.id,
                datetime.now(UTC) + timedelta(days=1),
                sync_redis=False,
                commit=True,
            )

            commits = {"n": 0}
            original = db.session.commit

            def _spy_commit(*a, **k):
                commits["n"] += 1
                return original(*a, **k)

            with patch.object(db.session, "commit", side_effect=_spy_commit):
                update_token_last_used(tok, commit=False)
            assert commits["n"] == 0

            row = RefreshToken.query.filter_by(token_hash=_sha(tok)).first()
            assert row is not None
            assert row.last_used_at is not None


@pytest.mark.integration
class TestP0_6RealAuthRedis:
    """Nécessite AUTH_REDIS_URL → instance noeviction (redis_auth_test)."""

    @pytest.fixture(autouse=True)
    def _require_real_auth_redis(self, app, monkeypatch):
        url = (os.getenv("AUTH_REDIS_URL") or "").strip()
        general = (os.getenv("REDIS_URL") or "").strip()
        if not url:
            pytest.skip("AUTH_REDIS_URL non défini")

        monkeypatch.setattr(redis, "from_url", _real_redis_from_url)
        reset_auth_redis_client()
        with app.app_context():
            app.config["AUTH_REDIS_URL"] = url
            if general:
                app.config["REDIS_URL"] = general
            try:
                client = get_auth_redis(force_new=True)
                assert client.ping() is True
                assert not isinstance(client.ttl("__missing__"), MagicMock)
            except Exception as exc:
                pytest.skip(f"Redis auth injoignable: {exc}")
        yield
        reset_auth_redis_client()

    def test_restart_preserves_current_and_previous(self, app):
        with app.app_context():
            from security.refresh_redis_rotation import (
                RedisRefreshState,
                classify_refresh_in_redis,
                publish_refresh_redis,
                rotate_refresh_redis,
            )

            r0 = f"r0-{uuid.uuid4()}"
            r1 = f"r1-{uuid.uuid4()}"
            uid = 424242
            publish_refresh_redis(uid, r0, ttl_seconds=600)
            rotate_refresh_redis(
                uid, r0, r1, ttl_seconds=600, session_id=str(uuid.uuid4())
            )
            assert (
                classify_refresh_in_redis(r1, user_id=uid).state
                == RedisRefreshState.CURRENT
            )
            assert (
                classify_refresh_in_redis(r0, user_id=uid).state
                == RedisRefreshState.PREVIOUS_WITHIN_GRACE
            )
            ttl_before = int(get_auth_redis().ttl(f"refresh_previous:{_sha(r0)}"))

            reset_auth_redis_client()
            client = get_auth_redis(force_new=True)
            try:
                client.bgrewriteaof()
            except Exception:
                pass
            reset_auth_redis_client()
            assert (
                classify_refresh_in_redis(r1, user_id=uid).state
                == RedisRefreshState.CURRENT
            )
            assert (
                classify_refresh_in_redis(r0, user_id=uid).state
                == RedisRefreshState.PREVIOUS_WITHIN_GRACE
            )
            ttl_after = int(get_auth_redis().ttl(f"refresh_previous:{_sha(r0)}"))
            assert ttl_before > 0
            assert ttl_after > 0
            assert abs(ttl_before - ttl_after) < 30

    def test_general_pressure_does_not_evict_auth_keys(self, app):
        general_url = (os.getenv("REDIS_URL") or "").strip()
        auth_url = (os.getenv("AUTH_REDIS_URL") or "").strip()
        if not general_url or not auth_url or general_url == auth_url:
            pytest.skip("REDIS_URL et AUTH_REDIS_URL distincts requis")

        with app.app_context():
            from security.refresh_redis_rotation import (
                RedisRefreshState,
                classify_refresh_in_redis,
                publish_refresh_redis,
            )

            # Assurer de la tête sur auth (tests précédents peuvent saturer)
            auth = get_auth_redis()
            try:
                auth.config_set("maxmemory", str(2 * 1024 * 1024))
            except Exception:
                pass

            r0 = f"r0-press-{uuid.uuid4()}"
            publish_refresh_redis(999001, r0, ttl_seconds=600)
            assert (
                classify_refresh_in_redis(r0, user_id=999001).state
                == RedisRefreshState.CURRENT
            )

            gen = _real_redis_from_url(general_url, decode_responses=True)
            for i in range(500):
                try:
                    gen.setex(f"cache_pressure:{i}", 60, "x" * 1024)
                except Exception:
                    break

            assert (
                classify_refresh_in_redis(r0, user_id=999001).state
                == RedisRefreshState.CURRENT
            )

    def test_auth_saturation_rejects_write_not_evict(self, app, monkeypatch):
        monkeypatch.setenv("REFRESH_FAIL_CLOSED", "true")
        with app.app_context():
            from security.refresh_redis_rotation import (
                RedisRefreshState,
                classify_refresh_in_redis,
                publish_refresh_redis,
            )
            from services.security.authentication import RefreshTokenService

            client = get_auth_redis()
            maxmem = int(client.info(section="memory").get("maxmemory") or 0)
            if maxmem <= 0 or maxmem > 8 * 1024 * 1024:
                pytest.skip("maxmemory auth trop grand pour saturer en test")

            # Isoler : vider les clés de fill précédentes, garder une tête minimale
            try:
                client.config_set("maxmemory", str(maxmem))
                for key in client.scan_iter(match="fill:*", count=500):
                    client.delete(key)
            except Exception:
                pass

            r0 = f"r0-sat-{uuid.uuid4()}"
            publish_refresh_redis(999002, r0, ttl_seconds=600)
            assert (
                classify_refresh_in_redis(r0, user_id=999002).state
                == RedisRefreshState.CURRENT
            )

            i = 0
            oom_hit = False
            try:
                while i < 100000:
                    try:
                        client.set(f"fill:{i}", "y" * 2048)
                    except Exception as exc:
                        if is_redis_oom_error(exc):
                            oom_hit = True
                            break
                        raise
                    i += 1

                assert oom_hit, "OOM non atteint — baisser maxmemory auth test"
                assert (
                    classify_refresh_in_redis(r0, user_id=999002).state
                    == RedisRefreshState.CURRENT
                )
                used = int(client.info(section="memory").get("used_memory") or 0)
                client.config_set("maxmemory", str(max(used, 1)))
                svc = RefreshTokenService()
                with pytest.raises(RefreshStoreUnavailableError):
                    svc.store_token(999003, f"new-{uuid.uuid4()}", ttl_seconds=60)

                stats = client.info(section="stats")
                assert int(stats.get("evicted_keys") or 0) == 0
            finally:
                try:
                    for key in client.scan_iter(match="fill:*", count=500):
                        client.delete(key)
                    client.config_set("maxmemory", str(maxmem))
                except Exception:
                    pass

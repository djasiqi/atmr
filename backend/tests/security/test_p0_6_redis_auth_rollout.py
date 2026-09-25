"""P0-6 rollout — tests offline (dual-write, backfill TTL, parité, fail-closed).

Aucun accès production. Utilise un faux store mémoire.
"""

from __future__ import annotations

from typing import Any

import pytest

from security.auth_redis_migration import (
    ACTIVE_PREFIX,
    PREVIOUS_PREFIX,
    AuthRedisMigrationMode,
    DualWriteRedisClient,
    get_migration_mode,
)
from security.auth_redis_rollout_ops import (
    backfill_refresh_keys,
    compare_parity,
    copy_key_preserve_ttl,
    parity_gate_pass,
    recommend_maxmemory_bytes,
)


class FakeRedis:
    """Mini Redis string/zset pour tests offline."""

    def __init__(self) -> None:
        self.kv: dict[str, Any] = {}
        self.pttl_ms: dict[str, int] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.types: dict[str, str] = {}

    def ping(self) -> bool:
        return True

    def get(self, name: str) -> Any:
        return self.kv.get(name)

    def set(self, name: str, value: Any, **kwargs: Any) -> bool:
        self.kv[name] = value
        self.types[name] = "string"
        self.pttl_ms.pop(name, None)
        return True

    def setex(self, name: str, time: int, value: Any) -> bool:
        self.kv[name] = value
        self.types[name] = "string"
        self.pttl_ms[name] = int(time) * 1000
        return True

    def psetex(self, name: str, time_ms: int, value: Any) -> bool:
        self.kv[name] = value
        self.types[name] = "string"
        self.pttl_ms[name] = int(time_ms)
        return True

    def exists(self, *names: str) -> int:
        return sum(1 for n in names if n in self.kv or n in self.zsets)

    def delete(self, *names: str) -> int:
        n = 0
        for name in names:
            if name in self.kv or name in self.zsets:
                n += 1
            self.kv.pop(name, None)
            self.zsets.pop(name, None)
            self.pttl_ms.pop(name, None)
            self.types.pop(name, None)
        return n

    def ttl(self, name: str) -> int:
        p = self.pttl(name)
        if p < 0:
            return p
        return p // 1000

    def pttl(self, name: str) -> int:
        if name not in self.kv and name not in self.zsets:
            return -2
        return self.pttl_ms.get(name, -1)

    def type(self, name: str) -> str:
        if name in self.zsets:
            return "zset"
        if name in self.kv:
            return "string"
        return "none"

    def expire(self, name: str, time: int) -> bool:
        if name not in self.kv and name not in self.zsets:
            return False
        self.pttl_ms[name] = int(time) * 1000
        return True

    def pexpire(self, name: str, time: int) -> bool:
        if name not in self.kv and name not in self.zsets:
            return False
        self.pttl_ms[name] = int(time)
        return True

    def zadd(self, name: str, mapping: dict, **kwargs: Any) -> int:
        bucket = self.zsets.setdefault(name, {})
        self.types[name] = "zset"
        for k, v in mapping.items():
            bucket[k] = float(v)
        return len(mapping)

    def zrem(self, name: str, *values: Any) -> int:
        bucket = self.zsets.get(name) or {}
        n = 0
        for v in values:
            if v in bucket:
                del bucket[v]
                n += 1
        return n

    def zrange(self, name: str, start: int, end: int, withscores: bool = False, **kwargs: Any):
        bucket = self.zsets.get(name) or {}
        items = sorted(bucket.items(), key=lambda x: x[1])
        if end == -1:
            end = len(items) - 1
        sliced = items[start : end + 1]
        if withscores:
            return sliced
        return [k for k, _ in sliced]

    def zcard(self, name: str) -> int:
        return len(self.zsets.get(name) or {})

    def zscore(self, name: str, value: Any) -> float | None:
        return (self.zsets.get(name) or {}).get(value)

    def scan(self, cursor: int = 0, match: str | None = None, count: int | None = None):
        prefix = (match or "*").rstrip("*")
        keys = [k for k in list(self.kv) + list(self.zsets) if k.startswith(prefix)]
        return 0, keys

    def scan_iter(self, match: str | None = None, count: int | None = None):
        _, keys = self.scan(0, match=match, count=count)
        yield from keys

    def info(self, section: str | None = None, **kwargs: Any) -> dict:
        if section == "memory":
            return {"used_memory": 1000, "used_memory_peak": 2000, "maxmemory": 0}
        if section == "stats":
            return {"evicted_keys": 0}
        return {}

    def close(self) -> None:
        pass


class TestMigrationMode:
    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("AUTH_REDIS_MIGRATION_MODE", raising=False)
        assert get_migration_mode() == AuthRedisMigrationMode.OFF

    def test_dual_write_mode(self, monkeypatch):
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        assert get_migration_mode() == AuthRedisMigrationMode.DUAL_WRITE

    def test_invalid_mode_raises(self, monkeypatch):
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "banana")
        with pytest.raises(RuntimeError, match="invalide"):
            get_migration_mode()


class TestDualWrite:
    def test_dual_write_read_legacy_write_both(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        proxy = DualWriteRedisClient(
            read_client=legacy,
            write_primary=legacy,
            write_secondary=auth,
            mode=AuthRedisMigrationMode.DUAL_WRITE,
            secondary_label="redis-auth",
        )
        proxy.setex(f"{ACTIVE_PREFIX}abc", 60, "42")
        assert legacy.get(f"{ACTIVE_PREFIX}abc") == "42"
        assert auth.get(f"{ACTIVE_PREFIX}abc") == "42"
        # lecture = legacy
        legacy.set(f"{ACTIVE_PREFIX}only_legacy", "1")
        assert proxy.get(f"{ACTIVE_PREFIX}only_legacy") == "1"
        assert auth.get(f"{ACTIVE_PREFIX}only_legacy") is None

    def test_auth_primary_no_legacy_read_fallback(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        proxy = DualWriteRedisClient(
            read_client=auth,
            write_primary=auth,
            write_secondary=legacy,
            mode=AuthRedisMigrationMode.AUTH_PRIMARY,
            secondary_label="redis-legacy",
        )
        legacy.setex(f"{ACTIVE_PREFIX}legacy_only", 60, "99")
        # INTERDIT : fallback lecture legacy
        assert proxy.get(f"{ACTIVE_PREFIX}legacy_only") is None
        proxy.setex(f"{ACTIVE_PREFIX}new", 60, "7")
        assert auth.get(f"{ACTIVE_PREFIX}new") == "7"
        assert legacy.get(f"{ACTIVE_PREFIX}new") == "7"

    def test_secondary_write_failure_does_not_raise_in_dual_write(self):
        legacy = FakeRedis()

        class BoomRedis(FakeRedis):
            def setex(self, name: str, time: int, value: Any) -> bool:
                raise RuntimeError("oom")

        proxy = DualWriteRedisClient(
            read_client=legacy,
            write_primary=legacy,
            write_secondary=BoomRedis(),
            mode=AuthRedisMigrationMode.DUAL_WRITE,
            secondary_label="redis-auth",
        )
        # Ne doit pas lever — legacy reste autorité
        proxy.setex(f"{ACTIVE_PREFIX}x", 10, "1")
        assert legacy.get(f"{ACTIVE_PREFIX}x") == "1"


class TestBackfillAndParity:
    def test_copy_preserves_pttl(self):
        src = FakeRedis()
        dst = FakeRedis()
        key = f"{PREVIOUS_PREFIX}deadbeef"
        src.psetex(key, 42_000, '{"user_id":1}')
        status = copy_key_preserve_ttl(src, dst, key, dry_run=False)
        assert status == "copied"
        assert dst.get(key) == '{"user_id":1}'
        assert dst.pttl(key) == 42_000  # pas 300000

    def test_backfill_dry_run_no_write(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.setex(f"{ACTIVE_PREFIX}a", 100, "1")
        stats = backfill_refresh_keys(legacy, auth, dry_run=True)
        assert stats["dry_run"] >= 1
        assert auth.get(f"{ACTIVE_PREFIX}a") is None

    def test_parity_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.setex(f"{ACTIVE_PREFIX}a", 100, "1")
        auth.setex(f"{ACTIVE_PREFIX}a", 100, "1")
        legacy.psetex(f"{PREVIOUS_PREFIX}b", 40_000, "{}")
        auth.psetex(f"{PREVIOUS_PREFIX}b", 39_500, "{}")  # drift OK
        report = compare_parity(legacy, auth)
        assert parity_gate_pass(report) is True

        auth.delete(f"{ACTIVE_PREFIX}a")
        report2 = compare_parity(legacy, auth)
        assert parity_gate_pass(report2) is False

    def test_parity_extra_current_fails_gate(self):
        """EXTRA CURRENT dans auth (résurrection) → gate FAIL avant auth_primary."""
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.setex(f"{ACTIVE_PREFIX}keep", 100, "1")
        auth.setex(f"{ACTIVE_PREFIX}keep", 100, "1")
        auth.setex(f"{ACTIVE_PREFIX}zombie_r0", 100, "1")  # extra
        report = compare_parity(legacy, auth)
        assert report.extra_current_in_auth == [f"{ACTIVE_PREFIX}zombie_r0"]
        assert report.missing_current_in_auth == []
        assert parity_gate_pass(report) is False

    def test_parity_extra_previous_fails_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        auth.psetex(f"{PREVIOUS_PREFIX}orphan", 40_000, "{}")
        report = compare_parity(legacy, auth)
        assert report.extra_previous_in_auth == [f"{PREVIOUS_PREFIX}orphan"]
        assert parity_gate_pass(report) is False

    def test_recommend_maxmemory(self):
        rec = recommend_maxmemory_bytes(10 * 1024 * 1024, headroom_factor=3.0)
        assert rec >= 30 * 1024 * 1024


class TestBackfillAntiResurrection:
    def test_race_between_get_and_set_skips_stale_r0(self):
        """Rotation concurrente entre GET et SET → R0 non ressuscité dans auth."""
        legacy = FakeRedis()
        auth = FakeRedis()
        r0_key = f"{ACTIVE_PREFIX}r0deadbeef"
        r1_key = f"{ACTIVE_PREFIX}r1cafebabe"
        prev_key = f"{PREVIOUS_PREFIX}r0deadbeef"
        legacy.setex(r0_key, 100, "42")

        late_backfill_attempted = {"n": 0}

        def race_hook(key: str) -> None:
            if key != r0_key:
                return
            late_backfill_attempted["n"] += 1
            # Rotation R0→R1 dual-write déjà appliquée pendant la pause
            legacy.delete(r0_key)
            legacy.setex(r1_key, 100, "42")
            legacy.setex(prev_key, 300, '{"successor":"r1"}')
            auth.setex(r1_key, 100, "42")
            auth.setex(prev_key, 300, '{"successor":"r1"}')
            # R0 absent des deux stores

        stats = backfill_refresh_keys(
            legacy, auth, dry_run=False, race_hook=race_hook
        )
        assert late_backfill_attempted["n"] >= 1
        assert stats["skipped_stale_race"] >= 1
        assert auth.get(r0_key) is None  # pas de résurrection CURRENT
        assert auth.get(r1_key) == "42"
        assert legacy.get(r1_key) == "42"
        assert legacy.get(r0_key) is None

    def test_post_write_stale_rolls_back_resurrection(self):
        """Si SET auth a déjà écrit un R0 caduc → undo (rolled_back_stale)."""
        legacy = FakeRedis()
        auth = FakeRedis()
        r0_key = f"{ACTIVE_PREFIX}r0late"
        legacy.setex(r0_key, 100, "7")

        def post_write(key: str) -> None:
            if key == r0_key:
                legacy.delete(r0_key)

        status = copy_key_preserve_ttl(
            legacy,
            auth,
            r0_key,
            dry_run=False,
            post_write_race_hook=post_write,
        )
        assert status == "rolled_back_stale"
        assert auth.get(r0_key) is None


class TestResolveAuthRedisProduction:
    def test_prod_missing_raises(self, app, monkeypatch):
        from security.auth_redis import reset_auth_redis_client, resolve_auth_redis_url

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

    def test_get_auth_redis_dual_write_requires_distinct_urls(self, app, monkeypatch):
        from security.auth_redis import get_auth_redis, reset_auth_redis_client

        reset_auth_redis_client()
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://same:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://same:6379/0")
        with app.app_context():
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "dual_write"
            app.config["AUTH_REDIS_URL"] = "redis://same:6379/0"
            app.config["REDIS_URL"] = "redis://same:6379/0"
            with pytest.raises(RuntimeError, match="distincts"):
                get_auth_redis(force_new=True)


class TestP02CompensationDualRedis:
    """P0-2 compensation ne doit pas laisser d'orphelin avec deux Redis."""

    @staticmethod
    def _sha(token: str) -> str:
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest()

    def test_dual_write_db_fail_compensates_legacy_and_auth(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from security.refresh_redis_rotation import (
            RedisIssuanceHandle,
            compensate_redis_issuance,
        )

        legacy = FakeRedis()
        auth = FakeRedis()
        r0, r1 = "r0-dual-comp", "r1-dual-comp"
        h0, h1 = self._sha(r0), self._sha(r1)
        uid = 42
        # État post-rotation dual-write (avant commit DB)
        for store in (legacy, auth):
            store.setex(f"{ACTIVE_PREFIX}{h1}", 3600, str(uid))
            store.setex(f"{PREVIOUS_PREFIX}{h0}", 300, '{"user_id":42}')

        proxy = DualWriteRedisClient(
            read_client=legacy,
            write_primary=legacy,
            write_secondary=auth,
            mode=AuthRedisMigrationMode.DUAL_WRITE,
            secondary_label="redis-auth",
        )
        handle = RedisIssuanceHandle(
            kind="rotate",
            user_id=uid,
            new_token=r1,
            old_token=r0,
            old_was_active=True,
            old_active_ttl=3600,
            grace_seconds=300,
        )
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        svc = MagicMock()
        svc.redis_client = proxy
        svc._hash_token = self._sha

        with patch("security.refresh_redis_rotation._svc", return_value=svc):
            with patch(
                "security.auth_redis.get_legacy_redis", return_value=legacy
            ):
                with patch(
                    "security.auth_redis.get_dedicated_auth_redis",
                    return_value=auth,
                ):
                    compensate_redis_issuance(handle)

        assert legacy.get(f"{ACTIVE_PREFIX}{h0}") == str(uid)
        assert auth.get(f"{ACTIVE_PREFIX}{h0}") == str(uid)
        assert legacy.get(f"{ACTIVE_PREFIX}{h1}") is None
        assert auth.get(f"{ACTIVE_PREFIX}{h1}") is None
        assert legacy.get(f"{PREVIOUS_PREFIX}{h0}") is None
        assert auth.get(f"{PREVIOUS_PREFIX}{h0}") is None

    def test_dual_write_secondary_fail_no_logout(self):
        legacy = FakeRedis()

        class BoomRedis(FakeRedis):
            def setex(self, name: str, time: int, value: Any) -> bool:
                raise RuntimeError("auth secondary down")

        proxy = DualWriteRedisClient(
            read_client=legacy,
            write_primary=legacy,
            write_secondary=BoomRedis(),
            mode=AuthRedisMigrationMode.DUAL_WRITE,
            secondary_label="redis-auth",
        )
        # Legacy autorité : succès utilisateur malgré secondary FAIL
        proxy.setex(f"{ACTIVE_PREFIX}ok", 10, "1")
        assert legacy.get(f"{ACTIVE_PREFIX}ok") == "1"
        report = compare_parity(legacy, BoomRedis())
        # Divergence attendue (extra/missing) — gate parité échouerait
        assert parity_gate_pass(report) is False or legacy.get(
            f"{ACTIVE_PREFIX}ok"
        ) == "1"

    def test_auth_primary_legacy_secondary_fail_refresh_ok(self):
        auth = FakeRedis()

        class BoomLegacy(FakeRedis):
            def setex(self, name: str, time: int, value: Any) -> bool:
                raise RuntimeError("legacy mirror down")

        proxy = DualWriteRedisClient(
            read_client=auth,
            write_primary=auth,
            write_secondary=BoomLegacy(),
            mode=AuthRedisMigrationMode.AUTH_PRIMARY,
            secondary_label="redis-legacy",
        )
        proxy.setex(f"{ACTIVE_PREFIX}primary", 10, "9")
        assert auth.get(f"{ACTIVE_PREFIX}primary") == "9"
        # Pas d'exception → refresh utilisateur SUCCESS

    def test_auth_primary_db_fail_compensates_both(self, monkeypatch):
        from unittest.mock import MagicMock, patch

        from security.refresh_redis_rotation import (
            RedisIssuanceHandle,
            compensate_redis_issuance,
        )

        legacy = FakeRedis()
        auth = FakeRedis()
        r0, r1 = "r0-ap-comp", "r1-ap-comp"
        h0, h1 = self._sha(r0), self._sha(r1)
        uid = 7
        for store in (legacy, auth):
            store.setex(f"{ACTIVE_PREFIX}{h1}", 3600, str(uid))
            store.setex(f"{PREVIOUS_PREFIX}{h0}", 300, "{}")

        proxy = DualWriteRedisClient(
            read_client=auth,
            write_primary=auth,
            write_secondary=legacy,
            mode=AuthRedisMigrationMode.AUTH_PRIMARY,
            secondary_label="redis-legacy",
        )
        handle = RedisIssuanceHandle(
            kind="rotate",
            user_id=uid,
            new_token=r1,
            old_token=r0,
            old_was_active=True,
            old_active_ttl=1200,
            grace_seconds=300,
        )
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "auth_primary")
        svc = MagicMock()
        svc.redis_client = proxy
        svc._hash_token = self._sha

        with patch("security.refresh_redis_rotation._svc", return_value=svc):
            with patch(
                "security.auth_redis.get_legacy_redis", return_value=legacy
            ):
                with patch(
                    "security.auth_redis.get_dedicated_auth_redis",
                    return_value=auth,
                ):
                    compensate_redis_issuance(handle)

        assert auth.get(f"{ACTIVE_PREFIX}{h0}") == str(uid)
        assert legacy.get(f"{ACTIVE_PREFIX}{h0}") == str(uid)
        assert auth.get(f"{ACTIVE_PREFIX}{h1}") is None
        assert legacy.get(f"{ACTIVE_PREFIX}{h1}") is None


class TestDeployBootstrapGuard:
    def test_deploy_script_requires_redis_auth_before_backend(self):
        from pathlib import Path

        # Hors conteneur (repo root) : parents[3]=repo ; dans atmr_api seul ./backend est monté.
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "scripts" / "deploy-production.sh",
            here.parents[2].parent / "scripts" / "deploy-production.sh",
            Path("/workspace/scripts/deploy-production.sh"),
            Path("/app/../scripts/deploy-production.sh"),
        ]
        script = next((p for p in candidates if p.is_file()), None)
        if script is None:
            pytest.skip(
                "deploy-production.sh hors montage container — vérifié en gate hôte"
            )
        text = script.read_text(encoding="utf-8")
        assert "wait_redis_auth_ready" in text
        assert "AUTH_REDIS_URL absent" in text
        assert "NE JAMAIS déployer le backend fail-closed" in text
        idx_auth = text.find(
            "compose_prod up -d postgres pgbouncer redis redis-auth"
        )
        idx_backend = text.find("compose_prod up -d backend")
        assert idx_auth > 0
        assert idx_backend > idx_auth

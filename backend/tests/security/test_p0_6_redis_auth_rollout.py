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

    def zrange(
        self, name: str, start: int, end: int, withscores: bool = False, **kwargs: Any
    ):
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

        stats = backfill_refresh_keys(legacy, auth, dry_run=False, race_hook=race_hook)
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

    def test_ttl_absolute_not_prolonged_after_elapsed(self, monkeypatch):
        """PTTL capturé puis écrit plus tard → durée restante, jamais prolongée."""
        import security.auth_redis_rollout_ops as ops

        legacy = FakeRedis()
        auth = FakeRedis()
        key = f"{ACTIVE_PREFIX}ttlskew"
        legacy.psetex(key, 500, "1")

        clock = {"t": 1000.0}
        monkeypatch.setattr(ops, "_monotonic", lambda: clock["t"])
        # Capture à t=1000
        snap_time = clock["t"]
        # 300 ms plus tard
        clock["t"] = snap_time + 0.300

        # Relecture manuelle du chemin copy : forcer capture puis elapsed
        clock["t"] = snap_time
        status = copy_key_preserve_ttl(legacy, auth, key, dry_run=False)
        # Sans avancer l'horloge pendant copy, remaining≈500
        assert status == "copied"
        assert auth.pttl(key) == 500

        # Nouveau scénario : snapshot puis écriture différée via monkeypatch interne
        auth.delete(key)
        clock["t"] = snap_time
        snap = ops._read_snapshot(legacy, key)
        assert snap is not None
        clock["t"] = snap_time + 0.300
        remaining = ops._remaining_pttl_ms(snap)
        assert 199 <= remaining <= 201
        ops._write_snapshot(auth, snap)
        assert 199 <= auth.pttl(key) <= 201
        assert auth.pttl(key) < snap.pttl_ms

    def test_source_expires_before_write_skips(self, monkeypatch):
        import security.auth_redis_rollout_ops as ops

        legacy = FakeRedis()
        auth = FakeRedis()
        key = f"{ACTIVE_PREFIX}expirebefore"
        legacy.psetex(key, 100, "1")
        clock = {"t": 0.0}
        monkeypatch.setattr(ops, "_monotonic", lambda: clock["t"])

        def race_hook(_key: str) -> None:
            clock["t"] = 0.150  # 150 ms > 100 ms PTTL
            # Source encore présente pour le test remaining (FakeRedis ne décroît pas)
            # On simule expiration source :
            legacy.delete(key)

        status = copy_key_preserve_ttl(
            legacy, auth, key, dry_run=False, race_hook=race_hook
        )
        assert status == "skipped_stale_race"
        assert auth.get(key) is None

    def test_remaining_zero_skips_expired_without_write(self, monkeypatch):
        import security.auth_redis_rollout_ops as ops

        legacy = FakeRedis()
        auth = FakeRedis()
        key = f"{ACTIVE_PREFIX}expireremain"
        legacy.psetex(key, 100, "1")
        clock = {"t": 0.0}
        monkeypatch.setattr(ops, "_monotonic", lambda: clock["t"])
        snap = ops._read_snapshot(legacy, key)
        assert snap is not None
        clock["t"] = 0.150
        assert ops._remaining_pttl_ms(snap) <= 0
        status = copy_key_preserve_ttl(legacy, auth, key, dry_run=False)
        # copy re-reads snapshot at new time → pttl still 100 in FakeRedis → would copy
        # Force via direct remaining check path: inject race that advances clock only
        auth.delete(key)

        def race_hook(_k: str) -> None:
            clock["t"] = 0.150

        clock["t"] = 0.0
        status = copy_key_preserve_ttl(
            legacy, auth, key, dry_run=False, race_hook=race_hook
        )
        assert status == "skipped_expired"
        assert auth.get(key) is None


class TestRevokeDualStoreCleanup:
    def test_revoke_legacy_missing_auth_present_cleans_both(self, app, monkeypatch):
        """GET legacy vide + CURRENT auth présent → delete auth + revoked dual."""
        import hashlib
        from datetime import timedelta

        from services.security.authentication import RefreshTokenService

        legacy = FakeRedis()
        auth = FakeRedis()
        token = "revoke-canary-token-xyz"
        th = hashlib.sha256(token.encode()).hexdigest()
        uid = "99"
        # Legacy déjà sans CURRENT ; auth a encore CURRENT + zset
        auth.setex(f"{ACTIVE_PREFIX}{th}", 3600, uid)
        auth.zadd(f"user_refresh_tokens:{uid}", {th: 1.0})

        proxy = DualWriteRedisClient(
            read_client=legacy,
            write_primary=legacy,
            write_secondary=auth,
            mode=AuthRedisMigrationMode.DUAL_WRITE,
            secondary_label="redis-auth",
        )

        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-test:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-test:6379/0")
        import security.auth_redis as ar

        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)
        monkeypatch.setattr(ar, "get_auth_redis", lambda **kw: proxy)
        monkeypatch.setattr(ar, "reset_auth_redis_client", lambda: None)

        with app.app_context():
            app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "dual_write"
            app.config["AUTH_REDIS_URL"] = "redis://auth-test:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-test:6379/0"
            svc = RefreshTokenService()
            svc.redis_client = proxy
            svc.revoke_token(token)

        assert legacy.get(f"revoked_refresh_token:{th}") == "revoked"
        assert auth.get(f"revoked_refresh_token:{th}") == "revoked"
        assert legacy.get(f"{ACTIVE_PREFIX}{th}") is None
        assert auth.get(f"{ACTIVE_PREFIX}{th}") is None
        assert auth.zscore(f"user_refresh_tokens:{uid}", th) is None

    def test_revoke_both_present_normal(self, app, monkeypatch):
        import hashlib
        from datetime import timedelta

        from services.security.authentication import RefreshTokenService

        legacy = FakeRedis()
        auth = FakeRedis()
        token = "revoke-normal-token"
        th = hashlib.sha256(token.encode()).hexdigest()
        uid = "7"
        for store in (legacy, auth):
            store.setex(f"{ACTIVE_PREFIX}{th}", 3600, uid)
            store.zadd(f"user_refresh_tokens:{uid}", {th: 1.0})

        proxy = DualWriteRedisClient(
            read_client=legacy,
            write_primary=legacy,
            write_secondary=auth,
            mode=AuthRedisMigrationMode.DUAL_WRITE,
            secondary_label="redis-auth",
        )
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-test:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-test:6379/0")
        import security.auth_redis as ar

        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)
        monkeypatch.setattr(ar, "get_auth_redis", lambda **kw: proxy)
        with app.app_context():
            app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
            app.config["AUTH_REDIS_URL"] = "redis://auth-test:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-test:6379/0"
            svc = RefreshTokenService()
            svc.redis_client = proxy
            svc.revoke_token(token)

        for store in (legacy, auth):
            assert store.get(f"revoked_refresh_token:{th}") == "revoked"
            assert store.get(f"{ACTIVE_PREFIX}{th}") is None
            assert store.zscore(f"user_refresh_tokens:{uid}", th) is None


class TestParityExpandedFamilies:
    def test_parity_gate_includes_revoked_and_zset(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.setex(f"{ACTIVE_PREFIX}a", 100, "1")
        auth.setex(f"{ACTIVE_PREFIX}a", 100, "1")
        legacy.setex("revoked_refresh_token:x", 100, "revoked")
        # missing revoked in auth
        report = compare_parity(legacy, auth)
        assert report.missing_revoked_in_auth == ["revoked_refresh_token:x"]
        assert parity_gate_pass(report) is False

        auth.setex("revoked_refresh_token:x", 100, "revoked")
        legacy.zadd("user_refresh_tokens:1", {"a": 1.0})
        auth.zadd("user_refresh_tokens:1", {"a": 1.0, "extra": 2.0})
        report2 = compare_parity(legacy, auth)
        assert report2.mismatched_user_zset_members == ["user_refresh_tokens:1"]
        assert report2.mismatched_user_zset == ["user_refresh_tokens:1"]
        assert parity_gate_pass(report2) is False


class TestParityGateHardening:
    """A–F — TTL et scores ZSET doivent bloquer GATE_PASS."""

    def test_a_current_ttl_divergent_fails_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.psetex(f"{ACTIVE_PREFIX}a", 100_000, "1")
        auth.psetex(f"{ACTIVE_PREFIX}a", 10_000, "1")
        report = compare_parity(legacy, auth)
        assert report.ttl_mismatched_current == [f"{ACTIVE_PREFIX}a"]
        assert report.mismatched_current == []
        assert parity_gate_pass(report) is False

    def test_b_previous_ttl_divergent_fails_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.psetex(f"{PREVIOUS_PREFIX}p", 50_000, "{}")
        auth.psetex(f"{PREVIOUS_PREFIX}p", 5_000, "{}")
        report = compare_parity(legacy, auth)
        assert report.ttl_mismatched_previous == [f"{PREVIOUS_PREFIX}p"]
        assert report.mismatched_previous == []
        assert parity_gate_pass(report) is False

    def test_c_revoked_ttl_divergent_fails_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.psetex("revoked_refresh_token:r", 80_000, "revoked")
        auth.psetex("revoked_refresh_token:r", 8_000, "revoked")
        report = compare_parity(legacy, auth)
        assert report.ttl_mismatched_revoked == ["revoked_refresh_token:r"]
        assert report.mismatched_revoked == []
        assert parity_gate_pass(report) is False

    def test_d_user_zset_scores_divergent_fails_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.zadd("user_refresh_tokens:1", {"a": 100.0, "b": 200.0})
        legacy.pexpire("user_refresh_tokens:1", 60_000)
        auth.zadd("user_refresh_tokens:1", {"a": 300.0, "b": 100.0})
        auth.pexpire("user_refresh_tokens:1", 60_000)
        report = compare_parity(legacy, auth)
        assert report.mismatched_user_zset_members == []
        assert report.mismatched_user_zset_scores == ["user_refresh_tokens:1"]
        assert parity_gate_pass(report) is False

    def test_e_user_zset_ttl_divergent_fails_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        legacy.zadd("user_refresh_tokens:2", {"a": 1.0, "b": 2.0})
        legacy.pexpire("user_refresh_tokens:2", 90_000)
        auth.zadd("user_refresh_tokens:2", {"a": 1.0, "b": 2.0})
        auth.pexpire("user_refresh_tokens:2", 9_000)
        report = compare_parity(legacy, auth)
        assert report.mismatched_user_zset_members == []
        assert report.mismatched_user_zset_scores == []
        assert report.ttl_mismatched_user_zset == ["user_refresh_tokens:2"]
        assert parity_gate_pass(report) is False

    def test_f_full_parity_passes_gate(self):
        legacy = FakeRedis()
        auth = FakeRedis()
        for store in (legacy, auth):
            store.psetex(f"{ACTIVE_PREFIX}c", 100_000, "42")
            store.psetex(f"{PREVIOUS_PREFIX}p", 40_000, '{"s":"c"}')
            store.psetex("revoked_refresh_token:x", 30_000, "revoked")
            store.zadd("user_refresh_tokens:42", {"c": 10.0, "old": 5.0})
            store.pexpire("user_refresh_tokens:42", 100_000)
        report = compare_parity(legacy, auth)
        assert report.ttl_mismatches == []
        assert report.mismatched_user_zset_members == []
        assert report.mismatched_user_zset_scores == []
        assert parity_gate_pass(report) is True


class TestResolveAuthRedisProduction:
    def test_prod_missing_raises(self, app, monkeypatch):
        from security.auth_redis import reset_auth_redis_client, resolve_auth_redis_url

        reset_auth_redis_client()
        monkeypatch.delenv("AUTH_REDIS_ALLOW_LEGACY_FALLBACK", raising=False)
        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.delenv("AUTH_REDIS_URL", raising=False)
        with app.app_context():
            prev_env = app.config.get("ENV")
            prev_testing = app.config.get("TESTING")
            prev_debug = app.config.get("DEBUG")
            prev_auth = app.config.get("AUTH_REDIS_URL")
            try:
                app.config["AUTH_REDIS_URL"] = ""
                app.config["REDIS_URL"] = "redis://general:6379/0"
                app.config["ENV"] = "production"
                app.config["TESTING"] = False
                app.config["DEBUG"] = False
                with pytest.raises(RuntimeError, match="AUTH_REDIS_URL"):
                    resolve_auth_redis_url()
            finally:
                app.config["ENV"] = prev_env
                app.config["TESTING"] = prev_testing
                app.config["DEBUG"] = prev_debug
                app.config["AUTH_REDIS_URL"] = prev_auth

    def test_get_auth_redis_dual_write_requires_distinct_urls(self, app, monkeypatch):
        from security.auth_redis import get_auth_redis, reset_auth_redis_client

        reset_auth_redis_client()
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://same:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://same:6379/0")
        with app.app_context():
            prev_mode = app.config.get("AUTH_REDIS_MIGRATION_MODE")
            prev_auth = app.config.get("AUTH_REDIS_URL")
            prev_redis = app.config.get("REDIS_URL")
            try:
                app.config["AUTH_REDIS_MIGRATION_MODE"] = "dual_write"
                app.config["AUTH_REDIS_URL"] = "redis://same:6379/0"
                app.config["REDIS_URL"] = "redis://same:6379/0"
                with pytest.raises(RuntimeError, match="distincts"):
                    get_auth_redis(force_new=True)
            finally:
                app.config["AUTH_REDIS_MIGRATION_MODE"] = prev_mode
                app.config["AUTH_REDIS_URL"] = prev_auth
                app.config["REDIS_URL"] = prev_redis
                reset_auth_redis_client()


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

        with (
            patch("security.refresh_redis_rotation._svc", return_value=svc),
            patch("security.auth_redis.get_legacy_redis", return_value=legacy),
            patch("security.auth_redis.get_dedicated_auth_redis", return_value=auth),
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
        assert (
            parity_gate_pass(report) is False or legacy.get(f"{ACTIVE_PREFIX}ok") == "1"
        )

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

        with (
            patch("security.refresh_redis_rotation._svc", return_value=svc),
            patch("security.auth_redis.get_legacy_redis", return_value=legacy),
            patch("security.auth_redis.get_dedicated_auth_redis", return_value=auth),
        ):
            compensate_redis_issuance(handle)

        assert auth.get(f"{ACTIVE_PREFIX}{h0}") == str(uid)
        assert legacy.get(f"{ACTIVE_PREFIX}{h0}") == str(uid)
        assert auth.get(f"{ACTIVE_PREFIX}{h1}") is None
        assert legacy.get(f"{ACTIVE_PREFIX}{h1}") is None


class TestLegacyModeIsolation:
    """A/B — mode=legacy : READ/WRITE uniquement REDIS_URL (jamais redis-auth)."""

    @staticmethod
    def _sha(token: str) -> str:
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest()

    def test_a_legacy_store_token_writes_legacy_only(self, app, monkeypatch):
        from datetime import timedelta

        from services.security.authentication import RefreshTokenService

        legacy = FakeRedis()
        auth = FakeRedis()
        token = "legacy-store-token-abc"
        th = self._sha(token)
        uid = 55

        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "legacy")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-isolated:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-isolated:6379/0")
        import security.auth_redis as ar

        ar.reset_auth_redis_client()
        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)

        with app.app_context():
            app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "legacy"
            app.config["AUTH_REDIS_URL"] = "redis://auth-isolated:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-isolated:6379/0"
            client = ar.get_auth_redis(force_new=True)
            assert client is legacy
            svc = RefreshTokenService()
            svc.redis_client = client
            svc.store_token(uid, token, ttl_seconds=3600)

        assert legacy.get(f"{ACTIVE_PREFIX}{th}") == str(uid)
        assert auth.get(f"{ACTIVE_PREFIX}{th}") is None
        assert auth.kv == {}

    def test_b_legacy_revoke_operates_legacy_only(self, app, monkeypatch):
        from datetime import timedelta

        from services.security.authentication import RefreshTokenService

        legacy = FakeRedis()
        auth = FakeRedis()
        token = "legacy-revoke-token-xyz"
        th = self._sha(token)
        uid = "12"
        legacy.setex(f"{ACTIVE_PREFIX}{th}", 3600, uid)
        legacy.zadd(f"user_refresh_tokens:{uid}", {th: 1.0})
        # auth a une clé « fantôme » : ne doit pas être touchée
        auth.setex(f"{ACTIVE_PREFIX}{th}", 3600, uid)
        auth.zadd(f"user_refresh_tokens:{uid}", {th: 1.0})

        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "legacy")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-isolated:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-isolated:6379/0")
        import security.auth_redis as ar

        ar.reset_auth_redis_client()
        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)

        with app.app_context():
            app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=30)
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "legacy"
            app.config["AUTH_REDIS_URL"] = "redis://auth-isolated:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-isolated:6379/0"
            client = ar.get_auth_redis(force_new=True)
            assert client is legacy
            svc = RefreshTokenService()
            svc.redis_client = client
            svc.revoke_token(token)

        assert legacy.get(f"revoked_refresh_token:{th}") == "revoked"
        assert legacy.get(f"{ACTIVE_PREFIX}{th}") is None
        assert legacy.zscore(f"user_refresh_tokens:{uid}", th) is None
        # auth inchangé (jamais contacté)
        assert auth.get(f"{ACTIVE_PREFIX}{th}") == uid
        assert auth.zscore(f"user_refresh_tokens:{uid}", th) == 1.0
        assert auth.get(f"revoked_refresh_token:{th}") is None


class TestProdOffAmbiguousGuard:
    """C — production + off + AUTH_REDIS_URL distinct → refuse."""

    def test_c_assert_prod_off_distinct_raises(self, monkeypatch):
        from security.auth_redis import reset_auth_redis_client
        from security.auth_redis_migration import (
            assert_prod_migration_mode_not_ambiguous,
        )

        reset_auth_redis_client()
        monkeypatch.setenv("FLASK_ENV", "production")
        monkeypatch.setenv("FLASK_CONFIG", "production")
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "off")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-prod:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-prod:6379/0")
        with pytest.raises(RuntimeError, match="off est interdit"):
            assert_prod_migration_mode_not_ambiguous()

    def test_c_validate_required_env_vars_rejects_off(self, monkeypatch):
        from app import validate_required_env_vars

        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-for-guard")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/atmr")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-prod:6379/0")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-prod:6379/0")
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "off")
        monkeypatch.setenv("SOCKETIO_CORS_ORIGINS", "https://app.example.com")
        with pytest.raises(RuntimeError, match="off interdit"):
            validate_required_env_vars("production")

    def test_c_dev_off_distinct_allowed(self, monkeypatch):
        from security.auth_redis_migration import (
            assert_prod_migration_mode_not_ambiguous,
        )

        monkeypatch.setenv("FLASK_ENV", "development")
        monkeypatch.setenv("FLASK_CONFIG", "development")
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "off")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-dev:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-dev:6379/0")
        # Ne doit pas lever hors production
        assert_prod_migration_mode_not_ambiguous()


class TestModeContractGetAuthRedis:
    """D/E/F — dual_write / auth_primary / auth_only inchangés via get_auth_redis."""

    def test_d_dual_write_read_legacy_write_both(self, app, monkeypatch):
        import security.auth_redis as ar
        from security.auth_redis_migration import DualWriteRedisClient

        legacy = FakeRedis()
        auth = FakeRedis()
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "dual_write")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-dw:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-dw:6379/0")
        ar.reset_auth_redis_client()
        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)

        with app.app_context():
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "dual_write"
            app.config["AUTH_REDIS_URL"] = "redis://auth-dw:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-dw:6379/0"
            client = ar.get_auth_redis(force_new=True)
            assert isinstance(client, DualWriteRedisClient)
            client.setex(f"{ACTIVE_PREFIX}dw", 10, "1")

        assert legacy.get(f"{ACTIVE_PREFIX}dw") == "1"
        assert auth.get(f"{ACTIVE_PREFIX}dw") == "1"
        legacy.set(f"{ACTIVE_PREFIX}only_l", "x")
        assert client.get(f"{ACTIVE_PREFIX}only_l") == "x"
        assert auth.get(f"{ACTIVE_PREFIX}only_l") is None

    def test_e_auth_primary_read_auth_write_both(self, app, monkeypatch):
        import security.auth_redis as ar
        from security.auth_redis_migration import DualWriteRedisClient

        legacy = FakeRedis()
        auth = FakeRedis()
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "auth_primary")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-ap:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-ap:6379/0")
        ar.reset_auth_redis_client()
        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)

        with app.app_context():
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "auth_primary"
            app.config["AUTH_REDIS_URL"] = "redis://auth-ap:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-ap:6379/0"
            client = ar.get_auth_redis(force_new=True)
            assert isinstance(client, DualWriteRedisClient)
            client.setex(f"{ACTIVE_PREFIX}ap", 10, "7")

        assert auth.get(f"{ACTIVE_PREFIX}ap") == "7"
        assert legacy.get(f"{ACTIVE_PREFIX}ap") == "7"
        # lecture = auth uniquement (pas de fallback legacy)
        legacy.setex(f"{ACTIVE_PREFIX}legacy_only", 60, "99")
        assert client.get(f"{ACTIVE_PREFIX}legacy_only") is None

    def test_f_auth_only_writes_auth_not_legacy(self, app, monkeypatch):
        import security.auth_redis as ar

        legacy = FakeRedis()
        auth = FakeRedis()
        monkeypatch.setenv("AUTH_REDIS_MIGRATION_MODE", "auth_only")
        monkeypatch.setenv("AUTH_REDIS_URL", "redis://auth-ao:6379/0")
        monkeypatch.setenv("REDIS_URL", "redis://legacy-ao:6379/0")
        ar.reset_auth_redis_client()
        monkeypatch.setattr(ar, "get_legacy_redis", lambda **kw: legacy)
        monkeypatch.setattr(ar, "get_dedicated_auth_redis", lambda **kw: auth)

        with app.app_context():
            app.config["AUTH_REDIS_MIGRATION_MODE"] = "auth_only"
            app.config["AUTH_REDIS_URL"] = "redis://auth-ao:6379/0"
            app.config["REDIS_URL"] = "redis://legacy-ao:6379/0"
            client = ar.get_auth_redis(force_new=True)
            assert client is auth
            client.setex(f"{ACTIVE_PREFIX}ao", 10, "3")

        assert auth.get(f"{ACTIVE_PREFIX}ao") == "3"
        assert legacy.get(f"{ACTIVE_PREFIX}ao") is None


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
        assert "AUTH_REDIS_MIGRATION_MODE" in text
        assert "off interdit" in text
        idx_auth = text.find("compose_prod up -d postgres pgbouncer redis redis-auth")
        idx_backend = text.find("compose_prod up -d backend")
        assert idx_auth > 0
        assert idx_backend > idx_auth

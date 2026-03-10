from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class DemoEnvironmentSnapshot:
    app_env: str
    demo_mode: bool
    database_url: str
    redis_url: str
    storage_bucket: str
    storage_prefix: str


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _database_name(database_url: str) -> str:
    parsed = urlparse(database_url)
    return parsed.path.lstrip("/").strip()


def _is_demo_db(database_url: str) -> bool:
    db_name = _database_name(database_url).lower()
    return bool(db_name) and (db_name.endswith("_demo") or "demo" in db_name)


def _is_demo_redis(redis_url: str) -> bool:
    parsed = urlparse(redis_url)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    return any(marker in host for marker in ("demo", "redis-demo")) or "demo" in path


def _is_demo_storage(bucket: str, prefix: str) -> bool:
    bucket_l = bucket.lower().strip()
    prefix_l = prefix.lower().strip()
    if bucket_l:
        if "-demo" in bucket_l or bucket_l.startswith("demo"):
            return True
    return prefix_l.startswith("demo/")


def build_demo_environment_snapshot() -> DemoEnvironmentSnapshot:
    app_env = (os.getenv("APP_ENV") or "").strip().lower()
    demo_mode = _is_truthy(os.getenv("DEMO_MODE"))
    database_url = (
        os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI") or ""
    ).strip()
    redis_url = (os.getenv("REDIS_URL") or "").strip()
    storage_bucket = (
        os.getenv("STORAGE_BUCKET")
        or os.getenv("S3_BUCKET")
        or os.getenv("DEMO_STORAGE_BUCKET")
        or ""
    ).strip()
    storage_prefix = (
        os.getenv("STORAGE_PREFIX")
        or os.getenv("S3_PREFIX")
        or os.getenv("DEMO_STORAGE_PREFIX")
        or ""
    ).strip()
    return DemoEnvironmentSnapshot(
        app_env=app_env,
        demo_mode=demo_mode,
        database_url=database_url,
        redis_url=redis_url,
        storage_bucket=storage_bucket,
        storage_prefix=storage_prefix,
    )


def enforce_demo_environment_or_raise(
    snapshot: DemoEnvironmentSnapshot,
    *,
    strict: bool = True,
) -> None:
    if snapshot.app_env != "demo":
        return

    if not snapshot.demo_mode:
        raise RuntimeError(
            "APP_ENV=demo exige DEMO_MODE=true pour éviter un boot ambigu."
        )

    if not snapshot.database_url:
        raise RuntimeError("APP_ENV=demo exige DATABASE_URL/SQLALCHEMY_DATABASE_URI.")
    if not _is_demo_db(snapshot.database_url):
        raise RuntimeError(
            "Refus de démarrage: la base demo doit cibler un nom suffixé `_demo`."
        )

    if not snapshot.redis_url:
        raise RuntimeError("APP_ENV=demo exige REDIS_URL dédié demo.")
    if not _is_demo_redis(snapshot.redis_url):
        raise RuntimeError(
            "Refus de démarrage: REDIS_URL demo doit pointer vers un hôte/url demo."
        )

    if strict and not _is_demo_storage(snapshot.storage_bucket, snapshot.storage_prefix):
        raise RuntimeError(
            "Refus de démarrage: storage demo invalide. Configurez un bucket `*-demo` "
            "ou un préfixe strict `demo/`."
        )


def block_sensitive_integrations_in_demo() -> dict[str, bool]:
    snapshot = build_demo_environment_snapshot()
    if snapshot.app_env != "demo":
        return {}

    blocked_features = {
        "payments": True,
        "sms": True,
        "external_webhooks": True,
        "insurance_apis": True,
        "accounting_exports": True,
    }
    return blocked_features


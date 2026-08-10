"""Tests construction DSN sûre et neutralisation Compose Kafka tracking."""

from __future__ import annotations

import os
import re
from hmac import compare_digest
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url


def _repo_root() -> Path:
    """Racine monorepo (compose) — /app en conteneur backend, ou parents[2] en checkout."""
    env_root = os.getenv("ATMR_REPO_ROOT")
    if env_root:
        candidate = Path(env_root)
        if (candidate / "docker-compose.production.yml").is_file():
            return candidate
    here = Path(__file__).resolve()
    for candidate in (
        here.parents[2],
        here.parents[1],
        Path("/repo"),
        Path("/app"),
        Path.cwd(),
    ):
        if (candidate / "docker-compose.production.yml").is_file():
            return candidate
        if (candidate.parent / "docker-compose.production.yml").is_file():
            return candidate.parent
    return here.parents[2]


REPO_ROOT = _repo_root()

_TRACKING_SERVICES = (
    "tracking-kafka-consumer",
    "tracking-processed-fanout",
    "kafka-dlq-consumer",
)

_EMPTY_DSN_KEYS = (
    "DATABASE_URL",
    "SQLALCHEMY_DATABASE_URI",
    "PRIMARY_DATABASE_URL",
    "REPLICA_DATABASE_URL",
    "REPLICA_DATABASE_URLS",
)


def test_build_database_url_safe_escapes_reserved_password_chars(monkeypatch):
    """Mot de passe avec caractères réservés URL — jamais via fichier .env."""
    fake_password = "p@ss:w/rd#x?y%z+ end"
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("POSTGRES_USER", "atmr")
    monkeypatch.setenv("POSTGRES_PASSWORD", fake_password)
    monkeypatch.setenv("POSTGRES_HOST", "pgbouncer")
    monkeypatch.setenv("POSTGRES_PORT", "6432")
    monkeypatch.setenv("POSTGRES_DB", "atmr")
    monkeypatch.setenv("POSTGRES_SSLMODE", "disable")

    from urllib.parse import quote_plus, unquote_plus

    from config import _build_database_url_safe

    raw_url = _build_database_url_safe()
    url = make_url(raw_url)

    assert url.drivername.startswith("postgresql")
    assert url.host == "pgbouncer"
    assert url.port == 6432
    assert url.database == "atmr"
    assert bool(url.username) is True
    assert bool(url.password) is True
    # make_url ne fait pas unquote_plus (+ vs espace) — vérifier l'encodage brut.
    userinfo = raw_url.split("://", 1)[1].split("@", 1)[0]
    _user, encoded_password = userinfo.split(":", 1)
    assert encoded_password == quote_plus(fake_password)
    assert compare_digest(unquote_plus(encoded_password), fake_password)


def test_pgbouncer_is_internal_host():
    from config import _is_internal_database_host

    assert _is_internal_database_host("pgbouncer") is True
    assert _is_internal_database_host("postgres") is True


def _service_environment_block(text: str, service: str) -> str | None:
    """Extrait le bloc environment: d'un service Compose (parse léger sans PyYAML)."""
    pattern = rf"(?m)^  {re.escape(service)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    service_body = match.group(1)
    env_match = re.search(
        r"(?m)^    environment:\n((?:^      .*\n)+)",
        service_body,
    )
    if not env_match:
        return None
    return env_match.group(1)


def _env_value(block: str, key: str) -> str | None:
    match = re.search(rf'(?m)^      {re.escape(key)}:\s*(?:"([^"]*)"|([^\n#]*))', block)
    if not match:
        return None
    return (
        match.group(1) if match.group(1) is not None else match.group(2) or ""
    ).rstrip()


@pytest.mark.parametrize(
    "compose_rel",
    [
        "docker-compose.kafka.yml",
        "docker-compose.kafka.dev.yml",
        "docker-compose.kafka.single.yml",
        "docker-compose.production.yml",
    ],
)
def test_tracking_kafka_services_neutralize_inherited_dsn(compose_rel: str):
    """Les trois services tracking n'interpolent plus le mot de passe dans l'URL."""
    compose_path = REPO_ROOT / compose_rel
    if not compose_path.is_file():
        pytest.skip(f"{compose_rel} absent du contexte de test ({REPO_ROOT})")

    text = compose_path.read_text(encoding="utf-8")
    found = [name for name in _TRACKING_SERVICES if f"  {name}:" in text]
    assert found, f"aucun service tracking dans {compose_rel}"

    for name in found:
        block = _service_environment_block(text, name)
        assert block is not None, f"{compose_rel}:{name} sans environment"
        assert _env_value(block, "POSTGRES_HOST") == "pgbouncer"
        port = _env_value(block, "POSTGRES_PORT")
        assert port in ("6432", '"6432"'), f"port inattendu: {port!r}"
        for key in _EMPTY_DSN_KEYS:
            val = _env_value(block, key)
            assert val is not None, f"{compose_rel}:{name} manque {key}"
            assert val in ("", '""'), f"{compose_rel}:{name}.{key} doit être vide"
        # Pas d'URL interpolée avec user/password Compose
        assert "postgresql+psycopg://${POSTGRES_USER" not in block
        assert "postgresql+psycopg://${POSTGRES_PASSWORD" not in block
        assert (
            re.search(r"postgresql\+psycopg://\$\{POSTGRES_(USER|PASSWORD)", block)
            is None
        )


def test_compose_dsn_fixture_documents_neutralized_urls():
    """Fixture embarquée (disponible dans le conteneur /app) — 5 URL vides."""
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "compose_dsn"
        / "expected_tracking_env.yml"
    )
    text = fixture.read_text(encoding="utf-8")
    for service in _TRACKING_SERVICES:
        assert f"{service}:" in text
    for key in _EMPTY_DSN_KEYS:
        assert f'{key}: ""' in text
    assert "POSTGRES_HOST: pgbouncer" in text
    assert 'POSTGRES_PORT: "6432"' in text
    assert "postgresql+psycopg://" not in text


def test_kraft_ingest_consumer_neutralizes_dsn():
    compose_path = REPO_ROOT / "docker-compose.kafka.kraft.yml"
    if not compose_path.is_file():
        pytest.skip("docker-compose.kafka.kraft.yml absent")
    text = compose_path.read_text(encoding="utf-8")
    block = _service_environment_block(text, "tracking-kafka-consumer")
    assert block is not None
    assert _env_value(block, "DATABASE_URL") in ("", '""')
    assert _env_value(block, "POSTGRES_HOST") == "pgbouncer"
    assert _env_value(block, "SQLALCHEMY_DATABASE_URI") in ("", '""')


def test_compose_config_json_fused_with_p0_hold_override(tmp_path):
    """Compose config --format json fusionné (prod + kafka + network + p0-hold)."""
    import json
    import shutil
    import subprocess

    required = (
        "docker-compose.production.yml",
        "docker-compose.kafka.yml",
        "docker-compose.kafka.atmr-network.yml",
        "docker-compose.kafka.p0-hold.yml",
    )
    for rel in required:
        if not (REPO_ROOT / rel).is_file():
            pytest.skip(f"{rel} absent")

    precomputed = os.getenv("ATMR_COMPOSE_CONFIG_JSON")
    if precomputed:
        data = json.loads(Path(precomputed).read_text(encoding="utf-8-sig"))
    else:
        if shutil.which("docker") is None:
            pytest.skip(
                "docker CLI indisponible "
                "(définir ATMR_COMPOSE_CONFIG_JSON pour un config pré-généré)"
            )

        env_file = tmp_path / "compose-p0-test.env"
        env_file.write_text(
            "\n".join(
                [
                    "POSTGRES_USER=atmr_test",
                    "POSTGRES_PASSWORD=fixture-not-a-real-secret",
                    "POSTGRES_DB=atmr_test",
                    "POSTGRES_SSLMODE=disable",
                    "REDIS_PASSWORD=fixture-redis",
                    "SECRET_KEY=fixture-secret-key-not-real",
                    "JWT_SECRET_KEY=fixture-jwt-secret-not-real",
                    "APP_ENCRYPTION_KEY_B64=Zml4dHVyZS1lbmNyeXB0aW9uLWtleS0zMg==",
                    "INTERNAL_SERVICE_TOKEN=fixture-internal-token",
                    "COMPOSE_PROJECT_NAME=atmr-p0-dsn-test",
                    "KAFKA_TOPIC_DRIVER_LOCATION_RAW=driver.location.raw.v2",
                    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED=driver.location.processed.v2",
                    "KAFKA_TOPIC_DRIVER_LOCATION_DLQ=driver.location.dlq.v2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        cmd = [
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "--profile",
            "kafka",
            "-f",
            "docker-compose.production.yml",
            "-f",
            "docker-compose.kafka.yml",
            "-f",
            "docker-compose.kafka.atmr-network.yml",
            "-f",
            "docker-compose.kafka.p0-hold.yml",
            "config",
            "--format",
            "json",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "COMPOSE_PROJECT_NAME": "atmr-p0-dsn-test"},
            check=False,
        )
        if proc.returncode != 0:
            pytest.fail(
                "docker compose config a échoué "
                f"(code={proc.returncode}): {proc.stderr[-2000:]}"
            )
        data = json.loads(proc.stdout)

    services = data.get("services") or {}

    for svc_name in _TRACKING_SERVICES:
        svc = services.get(svc_name)
        assert svc is not None, f"{svc_name} absent du config fusionné"
        env = svc.get("environment") or {}
        for key in _EMPTY_DSN_KEYS:
            assert key in env, f"{svc_name} manque {key}"
            assert env[key] in ("", None), (
                f"{svc_name}.{key} doit être vide, got {env[key]!r}"
            )
        assert env.get("POSTGRES_HOST") == "pgbouncer", svc_name
        assert str(env.get("POSTGRES_PORT")) == "6432", svc_name
        for val in env.values():
            if isinstance(val, str) and "postgresql+psycopg://" in val:
                assert "${POSTGRES_USER" not in val
                assert "${POSTGRES_PASSWORD" not in val
                assert "fixture-not-a-real-secret" not in val

    consumer_env = services["tracking-kafka-consumer"].get("environment") or {}
    assert consumer_env.get("TRACKING_PERSIST_WITH_OUTBOX") in ("true", True)
    assert consumer_env.get("TRACKING_INGEST_PERSIST_ENABLED") in ("true", True)
    assert consumer_env.get("TRACKING_INGEST_ALLOW_REPUBLISH_ONLY") in ("false", False)
    assert consumer_env.get("TRACKING_INGEST_SEEK_TO_END_ON_START") in ("false", False)
    assert consumer_env.get("TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE") in ("false", False)

    fanout_env = services["tracking-processed-fanout"].get("environment") or {}
    assert fanout_env.get("TRACKING_PROCESSED_FANOUT_ENABLED") in ("false", False)

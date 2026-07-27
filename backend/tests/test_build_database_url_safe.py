"""Tests construction DSN sûre et neutralisation Compose Kafka tracking."""

from __future__ import annotations

import re
from hmac import compare_digest
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url


def _repo_root() -> Path:
    """Racine monorepo (compose) — /app en conteneur backend, ou parents[2] en checkout."""
    here = Path(__file__).resolve()
    for candidate in (here.parents[2], here.parents[1], Path("/app"), Path.cwd()):
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
    return (match.group(1) if match.group(1) is not None else match.group(2) or "").rstrip()


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
        assert "${POSTGRES_PASSWORD" not in block
        assert "postgresql+psycopg://" not in block


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

#!/usr/bin/env python3
"""Anti-régression topics Kafka Compose (P0 GPS tracking).

1) Détecte les affectations littérales directes dans les fragments Kafka.
2) Vérifie que le compose fusionné production + kafka respecte les variables
   sentinelles (inline env), sans qu'un littéral ou un défaut les écrase.
3) Matrices image-only + override P0 (flags + absence de build).
4) Garde-fous statiques CI (v5, || true tracking, build: app dans kafka.yml).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMPOSE_FILES = (
    "docker-compose.kafka.yml",
    "docker-compose.kafka.kraft.yml",
    "docker-compose.kafka.dev.yml",
)

APP_SERVICES = (
    "tracking-kafka-consumer",
    "tracking-processed-fanout",
    "kafka-dlq-consumer",
)

# Affectation directe uniquement — ne pas matcher ${VAR:-driver.location.raw}
LITERAL_PATTERNS = (
    re.compile(r"^\s*KAFKA_TOPIC_DRIVER_LOCATION_RAW:\s*driver\.location\.raw\s*$"),
    re.compile(
        r"^\s*KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED:\s*driver\.location\.processed\s*$"
    ),
    re.compile(
        r"^\s*KAFKA_TOPIC_DRIVER_LOCATION_VALIDATED:\s*driver\.location\.processed\s*$"
    ),
    re.compile(r"^\s*KAFKA_TOPIC_DRIVER_LOCATION_DLQ:\s*driver\.location\.dlq\s*$"),
)

SENTINEL_RAW = "test.raw.v9"
SENTINEL_PROCESSED = "test.processed.v9"
SENTINEL_DLQ = "test.dlq.v9"

DOCKER_IMAGE = "djasiqi/atmr-backend"
DOCKER_TAG = "sha-deadbeef0123"

REQUIRED_DUMMY_ENV = {
    "POSTGRES_USER": "atmr_test",
    "POSTGRES_PASSWORD": "test-postgres",
    "POSTGRES_DB": "atmr_test",
    "POSTGRES_SSLMODE": "disable",
    "REDIS_PASSWORD": "test-redis",
    "REDIS_URL": "redis://redis:6379/0",
    "SECRET_KEY": "test-secret",
    "JWT_SECRET_KEY": "test-jwt",
    "APP_ENCRYPTION_KEY_B64": "dGVzdC1lbmNyeXB0aW9uLWtleS1iNjQ=",
    "INTERNAL_SERVICE_TOKEN": "test-internal-token",
    "MASTER_ENCRYPTION_KEY": "test-master-encryption-key",
    "COMPOSE_PROJECT_NAME": "atmr-ci-topic-sentinel",
    "DOCKER_IMAGE": DOCKER_IMAGE,
    "DOCKER_TAG": DOCKER_TAG,
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def pass_(msg: str) -> None:
    print(f"[PASS] {msg}")


def _write_env_file(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def check_no_literal_assignments() -> None:
    errors: list[str] = []
    for rel in COMPOSE_FILES:
        path = ROOT / rel
        if not path.is_file():
            fail(f"fichier manquant : {rel}")
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in LITERAL_PATTERNS:
                if pat.match(line):
                    errors.append(f"{rel}:{i}: {line.strip()}")
    if errors:
        fail(
            "affectation(s) littérale(s) interdite(s) :\n  "
            + "\n  ".join(errors)
        )
    pass_("aucune affectation littérale driver.location.{raw|processed|dlq}")


def _extract_service_block(config_yaml: str, service: str) -> str:
    """Extrait le bloc indenté d'un service top-level sous `services:`."""
    lines = config_yaml.splitlines()
    start = None
    service_indent = None
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if stripped.startswith(f"{service}:") and (
            stripped == f"{service}:" or stripped.startswith(f"{service}:")
        ):
            indent = len(line) - len(stripped)
            if indent <= 2:
                start = i
                service_indent = indent
                break
    if start is None:
        fail(f"service introuvable dans compose config : {service}")

    block: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip():
            block.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= (service_indent or 0) and line.lstrip(" ").split(":", 1)[0].strip():
            if ":" in line.lstrip(" "):
                break
        block.append(line)
    return "\n".join(block)


def _env_value_in_block(block: str, key: str) -> str | None:
    pat_map = re.compile(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", re.M)
    m = pat_map.search(block)
    if m:
        return m.group(1).strip().strip("\"'")
    pat_list = re.compile(rf"^\s*-\s*{re.escape(key)}=(.+?)\s*$", re.M)
    m2 = pat_list.search(block)
    if m2:
        return m2.group(1).strip().strip("\"'")
    return None


def _run_compose_config(extra_files: list[Path], env_values: dict[str, str]) -> dict:
    prod = ROOT / "docker-compose.production.yml"
    kafka = ROOT / "docker-compose.kafka.yml"
    net = ROOT / "docker-compose.kafka.atmr-network.yml"
    prod_env_path = ROOT / ".env.production"
    previous_prod_env: bytes | None = (
        prod_env_path.read_bytes() if prod_env_path.is_file() else None
    )

    env = os.environ.copy()
    env.update(env_values)
    env.pop("COMPOSE_ENV_FILES", None)

    try:
        _write_env_file(prod_env_path, env_values)
        with tempfile.TemporaryDirectory() as tmp:
            project_env_file = Path(tmp) / "compose-sentinel.env"
            _write_env_file(project_env_file, env_values)
            cmd = [
                "docker",
                "compose",
                "--env-file",
                str(project_env_file),
                "-f",
                str(prod),
                "-f",
                str(kafka),
                "-f",
                str(net),
            ]
            for extra in extra_files:
                cmd.extend(["-f", str(extra)])
            cmd.extend(["--profile", "kafka", "config", "--format", "json"])
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
            except FileNotFoundError:
                fail("docker compose introuvable — requis pour le test sentinelle")
            except subprocess.TimeoutExpired:
                fail("docker compose config timeout")
            if proc.returncode != 0:
                fail(
                    "docker compose config a échoué :\n"
                    + (proc.stderr or proc.stdout or "(pas de sortie)")
                )
            return json.loads(proc.stdout)
    finally:
        if previous_prod_env is None:
            try:
                prod_env_path.unlink(missing_ok=True)
            except OSError:
                pass
        else:
            prod_env_path.write_bytes(previous_prod_env)


def _service_env(svc: dict, key: str) -> str | None:
    env = svc.get("environment") or {}
    if isinstance(env, dict):
        val = env.get(key)
        if val is None:
            return None
        return str(val).strip().strip("\"'")
    if isinstance(env, list):
        prefix = f"{key}="
        for item in env:
            if isinstance(item, str) and item.startswith(prefix):
                return item[len(prefix) :].strip().strip("\"'")
    return None


def check_merged_compose_sentinels() -> None:
    env_values = {
        **REQUIRED_DUMMY_ENV,
        "KAFKA_TOPIC_DRIVER_LOCATION_RAW": SENTINEL_RAW,
        "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED": SENTINEL_PROCESSED,
        "KAFKA_TOPIC_DRIVER_LOCATION_DLQ": SENTINEL_DLQ,
    }
    data = _run_compose_config([], env_values)
    services = data.get("services") or {}
    expectations: dict[str, dict[str, str]] = {
        "tracking-kafka-consumer": {
            "KAFKA_TOPIC_DRIVER_LOCATION_RAW": SENTINEL_RAW,
            "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED": SENTINEL_PROCESSED,
            "KAFKA_TOPIC_DRIVER_LOCATION_VALIDATED": SENTINEL_PROCESSED,
            "KAFKA_TOPIC_DRIVER_LOCATION_DLQ": SENTINEL_DLQ,
        },
        "tracking-processed-fanout": {
            "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED": SENTINEL_PROCESSED,
        },
        "kafka-dlq-consumer": {
            "KAFKA_TOPIC_DRIVER_LOCATION_DLQ": SENTINEL_DLQ,
        },
    }
    for service, keys in expectations.items():
        svc = services.get(service)
        if not svc:
            fail(f"service introuvable : {service}")
        for key, expected in keys.items():
            got = _service_env(svc, key)
            if got != expected:
                fail(
                    f"{service}.{key} = {got!r}, attendu {expected!r} "
                    "(littéral ou défaut a écrasé la sentinelle)"
                )
        pass_(f"{service} : topics sentinelles OK")


def check_image_only_matrices() -> None:
    env_values = {
        **REQUIRED_DUMMY_ENV,
        "KAFKA_TOPIC_DRIVER_LOCATION_RAW": "driver.location.raw.v2",
        "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED": "driver.location.processed.v2",
        "KAFKA_TOPIC_DRIVER_LOCATION_DLQ": "driver.location.dlq.v2",
    }
    p0 = ROOT / "docker-compose.kafka.p0-hold.yml"
    matrices: list[tuple[str, list[Path]]] = [
        ("production+kafka+network", []),
        ("production+kafka+network+p0-hold", [p0]),
    ]
    expected_image = f"{DOCKER_IMAGE}:{DOCKER_TAG}"
    for name, extras in matrices:
        data = _run_compose_config(extras, env_values)
        services = data.get("services") or {}
        for svc_name in APP_SERVICES:
            svc = services.get(svc_name)
            if not svc:
                fail(f"{name}: service manquant {svc_name}")
            if "build" in svc:
                fail(f"{name}: {svc_name} contient encore build")
            image = svc.get("image")
            if image != expected_image:
                fail(f"{name}: {svc_name}.image={image!r} attendu {expected_image!r}")
        pass_(f"{name}: image-only OK ({expected_image})")

        if extras:
            ingest = services["tracking-kafka-consumer"]
            fanout = services["tracking-processed-fanout"]
            if _service_env(ingest, "TRACKING_INGEST_PERSIST_ENABLED") != "true":
                fail("P0: TRACKING_INGEST_PERSIST_ENABLED != true")
            if _service_env(ingest, "TRACKING_PERSIST_WITH_OUTBOX") != "true":
                fail("P0: TRACKING_PERSIST_WITH_OUTBOX != true (canary p0-hold)")
            if _service_env(fanout, "TRACKING_PROCESSED_FANOUT_ENABLED") != "false":
                fail("P0: TRACKING_PROCESSED_FANOUT_ENABLED != false")
            pass_("p0-hold flags OK")


def check_static_guards() -> None:
    deploy_yml = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
    kafka_yml = (ROOT / ".github/workflows/deploy-kafka.yml").read_text(encoding="utf-8")
    kafka_compose = (ROOT / "docker-compose.kafka.yml").read_text(encoding="utf-8")

    if re.search(r"DOCKER_TAG\s*.*\|\|\s*'v5'", deploy_yml) or "|| 'v5'" in deploy_yml:
        fail("deploy.yml contient encore un fallback v5")
    if "github.event.inputs.tag" in deploy_yml and "additional_tag" not in deploy_yml:
        fail("deploy.yml semble encore utiliser l'ancien input tag")
    pass_("deploy.yml : pas de fallback v5")

    if re.search(
        r"check-kafka-tracking-pipeline\.sh\s*\|\|\s*true",
        kafka_yml,
    ):
        fail("deploy-kafka.yml ignore encore check-kafka-tracking-pipeline avec || true")
    pass_("deploy-kafka.yml : pas de || true sur check tracking")

    # build: interdit sur les 3 services app GPS dans kafka.yml
    for svc in APP_SERVICES:
        # Cherche le bloc service puis un build: avant le prochain service top-level
        pat = re.compile(
            rf"^  {re.escape(svc)}:\n(.*?)(?=^  [a-zA-Z]|\Z)",
            re.M | re.S,
        )
        m = pat.search(kafka_compose)
        if not m:
            fail(f"service {svc} introuvable dans docker-compose.kafka.yml")
        block = m.group(1)
        if re.search(r"^\s+build:\s*$", block, re.M):
            fail(f"docker-compose.kafka.yml : {svc} contient encore build:")
    pass_("docker-compose.kafka.yml : pas de build: sur consumers GPS")

    build_override = ROOT / "docker-compose.kafka.build.yml"
    if not build_override.is_file():
        fail("docker-compose.kafka.build.yml manquant")
    pass_("docker-compose.kafka.build.yml présent")


def main() -> None:
    print("=== A2.1 anti-littéraux ===")
    check_no_literal_assignments()
    print("=== A2.2 compose fusionné sentinelles ===")
    check_merged_compose_sentinels()
    print("=== A2.3 matrices image-only + P0 ===")
    check_image_only_matrices()
    print("=== A2.4 garde-fous statiques ===")
    check_static_guards()
    print("Tous les checks Compose topics OK.")


if __name__ == "__main__":
    main()

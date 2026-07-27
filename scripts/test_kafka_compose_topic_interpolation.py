#!/usr/bin/env python3
"""Anti-régression topics Kafka Compose (P0 GPS tracking).

1) Détecte les affectations littérales directes dans les fragments Kafka.
2) Vérifie que le compose fusionné production + kafka respecte les variables
   sentinelles (inline env), sans qu'un littéral ou un défaut les écrase.
"""

from __future__ import annotations

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

REQUIRED_DUMMY_ENV = {
    "POSTGRES_PASSWORD": "test-postgres",
    "REDIS_PASSWORD": "test-redis",
    "SECRET_KEY": "test-secret",
    "JWT_SECRET_KEY": "test-jwt",
    "APP_ENCRYPTION_KEY_B64": "dGVzdC1lbmNyeXB0aW9uLWtleS1iNjQ=",
    "INTERNAL_SERVICE_TOKEN": "test-internal-token",
}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def pass_(msg: str) -> None:
    print(f"[PASS] {msg}")


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
    # Chercher "  service:" ou "service:" selon indentation compose config
    start = None
    service_indent = None
    for i, line in enumerate(lines):
        stripped = line.lstrip(" ")
        if stripped.startswith(f"{service}:") and (
            stripped == f"{service}:" or stripped.startswith(f"{service}:")
        ):
            # éviter les faux positifs (container_name etc.)
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
            # nouveau service / clé de même niveau
            if ":" in line.lstrip(" "):
                break
        block.append(line)
    return "\n".join(block)


def _env_value_in_block(block: str, key: str) -> str | None:
    # Formats possibles après `docker compose config` :
    #   KAFKA_TOPIC_DRIVER_LOCATION_RAW: test.raw.v9
    #   - KAFKA_TOPIC_DRIVER_LOCATION_RAW=test.raw.v9
    pat_map = re.compile(rf"^\s*{re.escape(key)}:\s*(.+?)\s*$", re.M)
    m = pat_map.search(block)
    if m:
        return m.group(1).strip().strip("\"'")
    pat_list = re.compile(rf"^\s*-\s*{re.escape(key)}=(.+?)\s*$", re.M)
    m2 = pat_list.search(block)
    if m2:
        return m2.group(1).strip().strip("\"'")
    return None


def check_merged_compose_sentinels() -> None:
    net = ROOT / "docker-compose.kafka.atmr-network.yml"
    prod = ROOT / "docker-compose.production.yml"
    kafka = ROOT / "docker-compose.kafka.yml"
    for p in (prod, kafka, net):
        if not p.is_file():
            fail(f"fichier manquant pour test fusionné : {p.name}")

    env = os.environ.copy()
    env.update(REQUIRED_DUMMY_ENV)
    env["KAFKA_TOPIC_DRIVER_LOCATION_RAW"] = SENTINEL_RAW
    env["KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED"] = SENTINEL_PROCESSED
    env["KAFKA_TOPIC_DRIVER_LOCATION_DLQ"] = SENTINEL_DLQ
    # Empêcher un .env local d'écraser les sentinelles
    env.pop("COMPOSE_ENV_FILES", None)

    with tempfile.TemporaryDirectory() as tmp:
        # Fichier env vide dédié — pas de .env.production qui réécrirait les topics
        dummy_env_file = Path(tmp) / "empty.env"
        dummy_env_file.write_text("", encoding="utf-8")

        cmd = [
            "docker",
            "compose",
            "--env-file",
            str(dummy_env_file),
            "-f",
            str(prod),
            "-f",
            str(kafka),
            "-f",
            str(net),
            "--profile",
            "kafka",
            "config",
        ]
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

        config = proc.stdout
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
            block = _extract_service_block(config, service)
            for key, expected in keys.items():
                got = _env_value_in_block(block, key)
                if got != expected:
                    fail(
                        f"{service}.{key} = {got!r}, attendu {expected!r} "
                        "(littéral ou défaut a écrasé la sentinelle)"
                    )
            pass_(f"{service} : topics sentinelles OK")


def main() -> None:
    print("=== A2.1 anti-littéraux ===")
    check_no_literal_assignments()
    print("=== A2.2 compose fusionné sentinelles ===")
    check_merged_compose_sentinels()
    print("Tous les checks Compose topics OK.")


if __name__ == "__main__":
    main()

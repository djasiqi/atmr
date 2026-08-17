#!/usr/bin/env bash
# Échoue immédiatement si le staging pointe vers la production.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

COMPOSE="docker-compose.staging.yml"
EXAMPLE="env.staging.example"
ENV_FILE=".env.staging"
COMPOSE_ONLY=0
if [[ "${1:-}" == "--compose-only" ]]; then
  COMPOSE_ONLY=1
fi

fail() {
  echo "PREFLIGHT FAIL: $*" >&2
  exit 1
}

pass() {
  echo "PREFLIGHT OK: $*"
}

[[ -f "$COMPOSE" ]] || fail "manque $COMPOSE"
[[ -f "$EXAMPLE" ]] || fail "manque $EXAMPLE"

# Noms / réseaux / hosts de production interdits dans les fichiers staging.
FORBIDDEN_PATTERNS=(
  "api.lirie.ch"
  "www.lirie.ch"
  "prometheus.lirie.ch"
  "grafana.lirie.ch"
  "lirie.ch"
  ".env.production"
  "docker-compose.production.yml"
  "atmr-postgres"
  "atmr-pgbouncer"
  "atmr-redis"
  "atmr-network"
  "traefik-network"
  "container_name: atmr-"
)

scan_file() {
  local file="$1"
  local content
  content="$(grep -v '^[[:space:]]*#' "$file" || true)"
  local pat
  for pat in "${FORBIDDEN_PATTERNS[@]}"; do
    if grep -F -q -- "$pat" <<<"$content"; then
      fail "$file contient la référence production interdite: $pat"
    fi
  done
}

scan_file "$COMPOSE"
scan_file "$EXAMPLE"
scan_file "monitoring/staging/prometheus.yml"

grep -q "^name: atmrstg" "$COMPOSE" || fail "$COMPOSE doit déclarer name: atmrstg"
grep -q "name: atmrstg_internal" "$COMPOSE" || fail "réseau staging atmrstg_internal manquant"
grep -q "name: atmrstg_pg_data" "$COMPOSE" || fail "volume postgres staging manquant"

# MODE défaut = off (jamais observe implicite)
if ! grep -q 'TRACKING_MISSION_FIREWALL_MODE: \${TRACKING_MISSION_FIREWALL_MODE:-off}' "$COMPOSE"; then
  fail "compose: TRACKING_MISSION_FIREWALL_MODE doit interpoler vers off"
fi
if grep -E 'TRACKING_MISSION_FIREWALL_MODE=\$\{TRACKING_MISSION_FIREWALL_MODE:-observe\}' "$EXAMPLE"; then
  fail "example: observe ne doit pas être la valeur par défaut"
fi
grep -q "^TRACKING_MISSION_FIREWALL_MODE=off" "$EXAMPLE" || fail "example: MODE=off attendu"
grep -q "^FLASK_CONFIG=production" "$EXAMPLE" || fail "example: FLASK_CONFIG=production attendu (pas StagingConfig)"
grep -q "^APP_ENV=staging" "$EXAMPLE" || fail "example: APP_ENV=staging attendu"

# Image jamais latest
if grep -Eiq 'STAGING_BACKEND_IMAGE=.*:latest' "$EXAMPLE"; then
  fail "example: latest interdit"
fi
grep -q "sha-d5694d8e7cec" "$EXAMPLE" || fail "example: tag sha-d5694d8e7cec manquant"

if [[ "$COMPOSE_ONLY" -eq 1 ]]; then
  pass "scan fichiers staging (compose-only) — 0 référence production"
  exit 0
fi

[[ -f "$ENV_FILE" ]] || fail "manque $ENV_FILE — lancer bash scripts/staging/init-env.sh"

scan_file "$ENV_FILE"

if grep -q "CHANGE_ME_GENERATE" "$ENV_FILE"; then
  fail "$ENV_FILE contient encore CHANGE_ME_GENERATE"
fi

IMAGE="$(grep -E '^STAGING_BACKEND_IMAGE=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '\r')"
[[ -n "$IMAGE" ]] || fail "STAGING_BACKEND_IMAGE vide"
[[ "$IMAGE" != *":latest"* ]] || fail "STAGING_BACKEND_IMAGE=latest interdit"
[[ "$IMAGE" == *"sha-d5694d8e7cec"* ]] || fail "STAGING_BACKEND_IMAGE doit pinner sha-d5694d8e7cec"

MODE="$(grep -E '^TRACKING_MISSION_FIREWALL_MODE=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '\r')"
[[ "$MODE" == "off" || "$MODE" == "observe" || "$MODE" == "enforce_mission" || "$MODE" == "strict" ]] || fail "MODE invalide: $MODE"

APPENV="$(grep -E '^APP_ENV=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '\r')"
[[ "$APPENV" == "staging" ]] || fail "APP_ENV doit être staging (vu: $APPENV)"
FLASKCFG="$(grep -E '^FLASK_CONFIG=' "$ENV_FILE" | tail -1 | cut -d= -f2- | tr -d '\r')"
[[ "$FLASKCFG" == "production" ]] || fail "FLASK_CONFIG doit être production (vu: $FLASKCFG)"

# Hosts DB/Redis/Kafka du fichier env ne doivent pas viser un FQDN prod
if grep -Ei 'DATABASE_URL=.*(lirie\.ch|amazonaws|azure|neon\.tech)' "$ENV_FILE"; then
  fail "DATABASE_URL staging pointe hors isolation"
fi
if grep -Ei 'REDIS_URL=.*(lirie\.ch|amazonaws|redis\.cloud)' "$ENV_FILE"; then
  fail "REDIS_URL staging pointe hors isolation"
fi
if grep -Ei 'KAFKA_BOOTSTRAP_SERVERS=.*(lirie\.ch|amazonaws|:9092,.+:9092,.+:9092)' "$ENV_FILE"; then
  fail "KAFKA_BOOTSTRAP_SERVERS staging suspect (prod/multi-broker externe)"
fi

pass "isolation staging — 0 référence production"
echo "MODE actuel dans .env.staging=$MODE (observe n'est autorisé qu'après baseline MODE=off)"

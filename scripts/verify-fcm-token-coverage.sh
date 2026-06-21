#!/usr/bin/env bash
# Vérification couverture FCM natif Android (chauffeurs).
#
# Usage :
#   ./scripts/verify-fcm-token-coverage.sh --report
#   ./scripts/verify-fcm-token-coverage.sh --driver-id 7514
#   ./scripts/verify-fcm-token-coverage.sh --driver-id 7514 --expect-fcm
#   ./scripts/verify-fcm-token-coverage.sh --driver-id 7514 --gate-json
#   ./scripts/verify-fcm-token-coverage.sh --android-expo-only
#
# Prod (SSH) :
#   docker exec atmr-backend-1 python scripts/verify_fcm_token_coverage.py --driver-id 7514
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PY_SCRIPT="scripts/verify_fcm_token_coverage.py"

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'atmr-backend-1'; then
  exec docker exec -i atmr-backend-1 python "$PY_SCRIPT" "$@"
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'atmr-atmr_api'; then
  exec docker exec -i atmr-atmr_api python "$PY_SCRIPT" "$@"
fi

if docker compose -f "$ROOT_DIR/docker-compose.yml" ps --services 2>/dev/null | grep -qx 'api'; then
  exec docker compose -f "$ROOT_DIR/docker-compose.yml" exec -T api python "$PY_SCRIPT" "$@"
fi

echo "ERREUR: container api introuvable (atmr-backend-1 ou docker compose api)." >&2
echo "Lance manuellement: docker exec <backend> python $PY_SCRIPT ..." >&2
exit 1

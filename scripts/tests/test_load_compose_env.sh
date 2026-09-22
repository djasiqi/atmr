#!/usr/bin/env bash
# Vérifie que load_compose_env_file n'exécute pas les valeurs « per … ».
set -o errexit -o nounset -o pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=/dev/null
source "${ROOT}/scripts/lib/load_compose_env.sh"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT
SENTINEL="${WORKDIR}/per_was_executed"
ENV_FILE="${WORKDIR}/compose.env"

cat >"$ENV_FILE" <<'EOF'
FOO=bar
INTERNAL_TRACKING_INGEST_RATE_LIMIT=100 per 15 minutes
QUOTED="hello per world"
# comment ignored
export BAZ=qux
EOF

# Si ce fichier était `source`, bash lancerait la commande `per`.
# On s'assure que rien de tel n'a lieu.
load_compose_env_file "$ENV_FILE"

if [ -e "$SENTINEL" ]; then
  echo "FAIL: une commande a été exécutée depuis la valeur env" >&2
  exit 1
fi
if [ "${FOO}" != "bar" ]; then
  echo "FAIL: FOO=${FOO}" >&2
  exit 1
fi
if [ "${INTERNAL_TRACKING_INGEST_RATE_LIMIT}" != "100 per 15 minutes" ]; then
  echo "FAIL: RATE_LIMIT='${INTERNAL_TRACKING_INGEST_RATE_LIMIT}'" >&2
  exit 1
fi
if [ "${QUOTED}" != "hello per world" ]; then
  echo "FAIL: QUOTED=${QUOTED}" >&2
  exit 1
fi
if [ "${BAZ}" != "qux" ]; then
  echo "FAIL: BAZ=${BAZ}" >&2
  exit 1
fi

# Contrôle négatif : source casserait exactement comme en prod.
if bash -c "set -e; source \"$ENV_FILE\"" 2>"${WORKDIR}/source.err"; then
  echo "FAIL: source aurait dû échouer sur 'per'" >&2
  exit 1
fi
if ! grep -q "per: command not found" "${WORKDIR}/source.err"; then
  echo "FAIL: source n'a pas produit 'per: command not found'" >&2
  cat "${WORKDIR}/source.err" >&2
  exit 1
fi

echo "PASS: load_compose_env_file (espaces conservés, aucune exécution shell)"

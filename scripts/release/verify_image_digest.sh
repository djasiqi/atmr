#!/usr/bin/env bash
# Vérifie que l'image runtime correspond au manifeste.
set -euo pipefail

MANIFEST="${1:?manifest.json requis}"
SERVICE="${2:?service compose requis (backend|ws-service)}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq requis" >&2
  exit 1
fi

if [ "$SERVICE" = "backend" ]; then
  EXPECTED="$(jq -r '.backend.reference' "$MANIFEST")"
elif [ "$SERVICE" = "ws-service" ]; then
  EXPECTED="$(jq -r '.ws.reference' "$MANIFEST")"
else
  echo "Service inconnu: $SERVICE" >&2
  exit 1
fi

CONTAINER_ID="$(docker compose -f docker-compose.production.yml ps -q "$SERVICE" | head -n1)"
if [ -z "$CONTAINER_ID" ]; then
  echo "Conteneur $SERVICE introuvable" >&2
  exit 1
fi

RUNTIME_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "$CONTAINER_ID" 2>/dev/null || true)"
IMAGE_REF="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER_ID")"

echo "Attendu (manifeste): ${EXPECTED}"
echo "Image config:        ${IMAGE_REF}"
echo "RepoDigest runtime:  ${RUNTIME_DIGEST}"

if [ "$IMAGE_REF" = "$EXPECTED" ] || [ "$RUNTIME_DIGEST" = "$EXPECTED" ]; then
  echo "✅ Digest runtime aligné sur le manifeste"
  exit 0
fi

# Accepter si le digest suffixe correspond
EXPECTED_DIGEST="${EXPECTED##*@}"
if [[ "$RUNTIME_DIGEST" == *"${EXPECTED_DIGEST}"* ]] || [[ "$IMAGE_REF" == *"${EXPECTED_DIGEST}"* ]]; then
  echo "✅ Digest runtime aligné (suffixe)"
  exit 0
fi

echo "❌ Digest runtime ≠ manifeste" >&2
exit 1

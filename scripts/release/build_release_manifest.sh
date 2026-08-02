#!/usr/bin/env bash
# Construit le manifeste de release (digests + SHA source).
set -euo pipefail

SOURCE_SHA="${SOURCE_SHA:?SOURCE_SHA requis}"
BACKEND_REPO="${BACKEND_REPO:?BACKEND_REPO requis}"
BACKEND_DIGEST="${BACKEND_DIGEST:?BACKEND_DIGEST requis}"
WS_REPO="${WS_REPO:?WS_REPO requis}"
WS_DIGEST="${WS_DIGEST:?WS_DIGEST requis}"
MIGRATION_HEAD="${MIGRATION_HEAD:-unknown}"
OUT="${OUT_FILE:-release-manifest.json}"

# Normaliser digest (avec ou sans préfixe sha256:)
norm_digest() {
  local d="$1"
  if [[ "$d" == sha256:* ]]; then
    echo "$d"
  else
    echo "sha256:${d}"
  fi
}

BACKEND_DIGEST_N="$(norm_digest "$BACKEND_DIGEST")"
WS_DIGEST_N="$(norm_digest "$WS_DIGEST")"

cat >"$OUT" <<EOF
{
  "schema_version": 1,
  "source_sha": "${SOURCE_SHA}",
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "migration_head": "${MIGRATION_HEAD}",
  "mobile_contract_version": "mobile-device-session-v1",
  "tracking_protocol_version": "v2",
  "backend": {
    "repository": "${BACKEND_REPO}",
    "digest": "${BACKEND_DIGEST_N}",
    "reference": "${BACKEND_REPO}@${BACKEND_DIGEST_N}"
  },
  "ws": {
    "repository": "${WS_REPO}",
    "digest": "${WS_DIGEST_N}",
    "reference": "${WS_REPO}@${WS_DIGEST_N}"
  }
}
EOF

echo "✅ Manifeste écrit: ${OUT}"
cat "$OUT"

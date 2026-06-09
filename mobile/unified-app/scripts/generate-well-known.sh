#!/usr/bin/env bash
# Génère assetlinks.json avec le SHA-256 Play App Signing (colons, majuscules).
# Usage: PLAY_APP_SIGNING_SHA256="AA:BB:..." ./scripts/generate-well-known.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$ROOT/public/.well-known"
SHA="${PLAY_APP_SIGNING_SHA256:-}"

if [[ -z "$SHA" ]]; then
  echo "ERROR: PLAY_APP_SIGNING_SHA256 requis (Play Console → App integrity → App signing key certificate)" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/assetlinks.json" <<EOF
[
  {
    "relation": ["delegate_permission/common.handle_all_urls"],
    "target": {
      "namespace": "android_app",
      "package_name": "ch.liri.operations",
      "sha256_cert_fingerprints": [
        "$SHA"
      ]
    }
  }
]
EOF

echo "Wrote $OUT_DIR/assetlinks.json"
echo "AASA inchangé : $OUT_DIR/apple-app-site-association"

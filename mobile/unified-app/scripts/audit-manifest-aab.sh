#!/usr/bin/env bash
# Audit STOP GATE Manifest sur un AAB (nécessite bundletool dans PATH).
# Usage: ./scripts/audit-manifest-aab.sh path/to/app.aab
set -euo pipefail

AAB="${1:-}"
if [[ -z "$AAB" || ! -f "$AAB" ]]; then
  echo "Usage: $0 path/to/app.aab" >&2
  exit 1
fi

if ! command -v bundletool >/dev/null 2>&1; then
  echo "ERROR: bundletool requis (https://github.com/google/bundletool)" >&2
  exit 1
fi

DUMP=$(bundletool dump manifest --bundle="$AAB")
echo "$DUMP"

FAIL=0

if ! echo "$DUMP" | grep -q "RECORD_AUDIO"; then
  echo "FAIL: RECORD_AUDIO absent du manifest (requis pour les messages vocaux)" >&2
  FAIL=1
else
  echo "OK: RECORD_AUDIO présent (messages vocaux)"
fi

if ! echo "$DUMP" | grep -q "foregroundServiceType"; then
  echo "WARN: foregroundServiceType non trouvé — vérifier manuellement" >&2
fi

if ! echo "$DUMP" | grep -q "app.lirie.ch"; then
  echo "FAIL: host app.lirie.ch absent des intent filters" >&2
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

echo "OK: audit manifest de base passé"

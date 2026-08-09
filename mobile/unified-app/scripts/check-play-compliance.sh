#!/usr/bin/env bash
# Echoue si requestPermissionsAsync apparait ailleurs que requestNotificationOsPermissions.ts (prod).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MATCHES=$(grep -R "requestPermissionsAsync(" src app \
  --include="*.ts" --include="*.tsx" \
  | grep -v test | grep -v ".test." || true)

COUNT=$(echo "$MATCHES" | grep -c "requestPermissionsAsync" || true)

if [[ "$COUNT" -ne 1 ]]; then
  echo "FAIL: attendu 1 occurrence prod de requestPermissionsAsync, trouve $COUNT" >&2
  echo "$MATCHES" >&2
  exit 1
fi

if ! echo "$MATCHES" | grep -q "requestNotificationOsPermissions.ts"; then
  echo "FAIL: requestPermissionsAsync doit etre dans requestNotificationOsPermissions.ts uniquement" >&2
  echo "$MATCHES" >&2
  exit 1
fi

echo "OK: requestPermissionsAsync - 1 site prod (requestNotificationOsPermissions.ts)"

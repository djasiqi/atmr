#!/usr/bin/env bash
# Gate CI N0 — Architecture Review (contrat + invariants + linter).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

echo "=== Architecture Review CI ==="

python scripts/architecture/check_tracking_contract.py

if [[ -d backend ]]; then
  docker compose run --rm --no-deps backend pytest \
    tests/architecture/ tests/contracts/ -q --tb=short 2>/dev/null \
    || pytest backend/tests/architecture/ backend/tests/contracts/ -q --tb=short
fi

echo "Architecture Review CI: OK"

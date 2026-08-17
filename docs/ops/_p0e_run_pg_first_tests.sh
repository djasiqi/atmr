#!/bin/sh
# run inside atmr-backend-1
set -e
cd /app
python - <<'PY'
import importlib.util
print("pytest", bool(importlib.util.find_spec("pytest")))
PY
mkdir -p /tmp/p0e_tests/services
# tests already scp'd beside this script
ls -la /tmp/p0e_tests/services/ || true
if python -c "import pytest" 2>/dev/null; then
  cd /tmp/p0e_tests
  PYTHONPATH=/app pytest -q services/test_p5b_pg_first_promotion.py services/test_location_db_persist_p01.py 2>&1 | tail -40
else
  echo "NO_PYTEST — running embedded safety gates"
  PYTHONPATH=/app python /tmp/p0e_pg_first_safety_gates.py
fi

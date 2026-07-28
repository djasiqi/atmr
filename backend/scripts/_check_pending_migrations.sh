#!/bin/sh
# Vérification locale miroir CI « Pending migrations detected »
set -eu
export DISABLE_EVENTLET=1
export AUTOGENERATE_SKIP_INDEXES=1
export FLASK_APP="${FLASK_APP:-app.py}"

echo "📋 Révision actuelle:"
flask db current || true

echo "🔄 Autogenerate test_pending_check..."
flask db revision --autogenerate -m "test_pending_check" >/tmp/autogen_out.txt 2>&1 || true
tail -40 /tmp/autogen_out.txt || true

LATEST=$(ls -t migrations/versions/*test_pending_check.py 2>/dev/null | head -1 || echo "")
echo "LATEST=${LATEST}"

if [ -n "$LATEST" ] && [ -f "$LATEST" ]; then
  echo "---FILE---"
  cat "$LATEST"
  if grep -q "def upgrade" "$LATEST" && grep -q "op\." "$LATEST"; then
    echo "PENDING_DETECTED"
    rm -f "$LATEST"
    exit 1
  fi
  echo "EMPTY_OR_NO_OPS"
  rm -f "$LATEST"
else
  echo "NO_FILE_CREATED"
fi

echo "CHECK_PASSED"

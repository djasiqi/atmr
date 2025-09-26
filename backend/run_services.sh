#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Charger .env proprement (pas de xargs)
set -a
[ -f ./.env ] && . ./.env
set +a

: "${CELERY_BROKER_URL:=redis://127.0.0.1:6379/0}"

echo "Starting Flask app..."
python app.py & FLASK_PID=$!

echo "Starting Celery worker..."
celery -A celery_app.celery -b "$CELERY_BROKER_URL" worker --loglevel=info & WORKER_PID=$!

echo "Starting Celery beat..."
celery -A celery_app.celery -b "$CELERY_BROKER_URL" beat --loglevel=info & BEAT_PID=$!

echo "Services started:"
echo "  Flask PID:  $FLASK_PID"
echo "  Worker PID: $WORKER_PID"
echo "  Beat PID:   $BEAT_PID"

wait

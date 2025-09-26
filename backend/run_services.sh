#!/bin/bash
cd "$(dirname "$0")"
export $(grep -v '^#' .env | xargs)
echo "Starting Flask app..."
python app.py &
FLASK_PID=$!
echo "Starting Celery worker..."
celery -A celery_app.celery worker --loglevel=info &
WORKER_PID=$!
echo "Starting Celery beat..."
celery -A celery_app.celery beat --loglevel=info &
BEAT_PID=$!
echo "Services started:"
echo "Flask PID: $FLASK_PID"
echo "Worker PID: $WORKER_PID"
echo "Beat PID: $BEAT_PID"
wait

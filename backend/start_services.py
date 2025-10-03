#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
BACKEND = ROOT  # ce script est dans backend

# Charger .env simplement
env_path = ROOT / '.env'
if env_path.exists():
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k, v)

broker = os.environ.get('CELERY_BROKER_URL', 'redis://127.0.0.1:6379/0')

print("Starting Flask app on port", os.environ.get('FLASK_RUN_PORT', '5000'))
flask = subprocess.Popen([sys.executable, 'app.py'], cwd=str(BACKEND))
time.sleep(2)

print("Starting Celery worker...")
worker = subprocess.Popen([
    sys.executable, '-m', 'celery', '-A', 'celery_app.celery', '-b', broker,
    'worker', '--loglevel=info'
], cwd=str(BACKEND))

print("Starting Celery beat...")
beat = subprocess.Popen([
    sys.executable, '-m', 'celery', '-A', 'celery_app.celery', '-b', broker,
    'beat', '--loglevel=info'
], cwd=str(BACKEND))

print("PIDs -> Flask:", flask.pid, " Worker:", worker.pid, " Beat:", beat.pid)
try:
    flask.wait()
finally:
    for p in (worker, beat):
        p.terminate()

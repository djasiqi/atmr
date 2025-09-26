#!/usr/bin/env python3
import os
import subprocess
import sys
import time

# Load environment variables
with open('.env', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

print("Environment loaded. Starting services...")

# Start Flask app
print("Starting Flask app on port 5000...")
flask_process = subprocess.Popen([
    sys.executable, 'app.py'
], cwd='/workspace/atmr/backend')

# Wait a bit for Flask to start
time.sleep(3)

# Start Celery worker
print("Starting Celery worker...")
celery_process = subprocess.Popen([
    sys.executable, '-m', 'celery', '-A', 'celery_app.celery', 'worker', '--loglevel=info'
], cwd='/workspace/atmr/backend')

# Start Celery beat
print("Starting Celery beat...")
beat_process = subprocess.Popen([
    sys.executable, '-m', 'celery', '-A', 'celery_app.celery', 'beat', '--loglevel=info'
], cwd='/workspace/atmr/backend')

print("Services started!")
print("Flask PID:", flask_process.pid)
print("Celery Worker PID:", celery_process.pid)
print("Celery Beat PID:", beat_process.pid)

try:
    flask_process.wait()
except KeyboardInterrupt:
    print("\nShutting down services...")
    flask_process.terminate()
    celery_process.terminate()
    beat_process.terminate()
    flask_process.wait()
    celery_process.wait()
    beat_process.wait()
    print("Services stopped.")

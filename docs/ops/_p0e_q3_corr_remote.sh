set -e
echo "=== containers ==="
docker ps --format "{{.Names}}"
echo "=== register/session around 13:26Z ==="
docker logs atmr-backend-1 --since "2026-08-17T13:25:50Z" --until "2026-08-17T13:27:00Z" 2>&1 | grep -E "trk_sess_17869731|session_conflict|register_tracking|tracking_sessions|/tracking/session|superseded|20135" | head -100 || true
echo "=== consumer ==="
docker logs atmr-consumer-1 --since "2026-08-17T13:25:50Z" --until "2026-08-17T13:27:00Z" 2>&1 | grep -E "session_conflict|lauam301|3zzbvuqa|gdnf3xtm|publish_realtime|superseded" | head -40 || true
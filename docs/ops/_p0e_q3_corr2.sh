set -e
echo "=== traefik access 13:26 ==="
docker logs traefik --since "2026-08-17T13:25:50Z" --until "2026-08-17T13:27:00Z" 2>&1 | grep -E "tracking/sessions|session_conflict|/driver/me/location" | head -80 || true
echo "=== ws-service ==="
docker logs atmr-ws-service --since "2026-08-17T13:25:50Z" --until "2026-08-17T13:27:00Z" 2>&1 | grep -E "session_conflict|force_tracking|reconnect|20135|driver_location_batch" | head -80 || true
echo "=== backend any level around register ==="
docker logs atmr-backend-1 --since "2026-08-17T13:25:50Z" --until "2026-08-17T13:27:00Z" 2>&1 | head -5
docker logs atmr-backend-1 --since "2026-08-17T13:25:50Z" --until "2026-08-17T13:27:00Z" 2>&1 | wc -l
echo "=== grep sessions path in backend all logs last 2h for register ==="
docker logs atmr-backend-1 --since "2026-08-17T13:20:00Z" --until "2026-08-17T13:35:00Z" 2>&1 | grep -iE "tracking/sessions|register_tracking_session|session_conflict" | head -50 || true
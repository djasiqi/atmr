set -e
echo "=== 409 / session_conflict HTTP 13:20-13:35 ==="
docker logs traefik --since "2026-08-17T13:20:00Z" --until "2026-08-17T13:35:00Z" 2>&1 | grep -E "409|/tracking/sessions" | head -80 || true
echo "=== ALL tracking/sessions POST that window ==="
docker logs traefik --since "2026-08-17T13:18:00Z" --until "2026-08-17T13:35:00Z" 2>&1 | grep "tracking/sessions" || true
echo "=== location 409 only ==="
docker logs traefik --since "2026-08-17T13:25:00Z" --until "2026-08-17T13:28:00Z" 2>&1 | grep ' /api/v1/driver/me/location ' | grep ' 409 ' | head -20 || echo "(none)"
echo "=== batch endpoint if any ==="
docker logs traefik --since "2026-08-17T13:25:00Z" --until "2026-08-17T13:28:00Z" 2>&1 | grep -iE "location/batch|locations|session_conflict" | head -40 || true
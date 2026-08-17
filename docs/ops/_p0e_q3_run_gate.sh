set -e
TS_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "GATE_START_UTC=$TS_START"
# Ensure gate script in container
docker cp /tmp/_p0e_session_stability_gate.py atmr-backend-1:/tmp/ 2>/dev/null || true
# health quick
echo "=== HEALTH ==="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED TRACKING_PERSIST_WITH_OUTBOX
docker inspect --format '{{.State.Health.Status}}' atmr-backend-1 2>/dev/null || echo backend_no_health
docker inspect --format '{{.State.Status}}' atmr-tracking-kafka-consumer-1
docker inspect --format '{{.State.Status}}' atmr-tracking-outbox-publisher-1
# Run gate 75s
export P0E_STABLE_SEC=75 P0E_MIN_NEW_DLE=3 P0E_POLL_SEC=5 P0E_DRIVER_ID=20135 P0E_MISSION_ID=38243
# Pass PG_FIRST from container env into script via docker exec -e
docker exec -e P0E_STABLE_SEC=75 -e P0E_MIN_NEW_DLE=3 -e P0E_POLL_SEC=5 \
  -e P0E_DRIVER_ID=20135 -e P0E_MISSION_ID=38243 \
  atmr-backend-1 python /tmp/_p0e_session_stability_gate.py
GATE_RC=$?
TS_END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "GATE_END_UTC=$TS_END GATE_RC=$GATE_RC"
echo "=== TRAEFIK POST sessions during window ==="
docker logs traefik --since "$TS_START" --until "$TS_END" 2>&1 | grep "tracking/sessions" || echo "(none)"
echo "=== TRAEFIK location 409 during window ==="
docker logs traefik --since "$TS_START" --until "$TS_END" 2>&1 | grep 'driver/me/location' | grep ' 409 ' || echo "(none)"
exit $GATE_RC
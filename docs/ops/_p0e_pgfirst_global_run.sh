set -euo pipefail
cd /srv/atmr
cp /tmp/_p0e_phase2_enable.sh /tmp/_p0e_enable_global.sh
cp /tmp/_p0e_phase2_rollback.sh /tmp/_p0e_rollback_global.sh
chmod +x /tmp/_p0e_enable_global.sh /tmp/_p0e_rollback_global.sh

echo "===== GLOBAL ENABLE PG_FIRST ====="
bash /tmp/_p0e_enable_global.sh

echo "===== HEALTH + LOGS SMOKE ====="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED TRACKING_PERSIST_WITH_OUTBOX
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED TRACKING_PERSIST_WITH_OUTBOX
docker logs atmr-tracking-kafka-consumer-1 --since 2m 2>&1 | grep -iE "Traceback|ERROR|promote" | tail -30 || echo "(no traceback/promote line)"

echo "===== OBS 180s ====="
docker cp /tmp/_p0e_pgfirst_obs.py atmr-backend-1:/tmp/
set +e
docker exec -e P0E_OBS_SEC=180 -e P0E_POLL_SEC=15 -e P0E_DRIVER_ID=20135 \
  atmr-backend-1 python /tmp/_p0e_pgfirst_obs.py
OBS_RC=$?
set -e
echo "OBS_RC=$OBS_RC"

echo "===== POST-OBS HEALTH ====="
docker inspect -f '{{.State.Health.Status}}' atmr-backend-1
docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1
docker inspect -f '{{.State.Status}}' atmr-tracking-outbox-publisher-1
docker logs atmr-tracking-kafka-consumer-1 --since 3m 2>&1 | grep -i Traceback | tail -10 || echo "(no traceback)"

if [ "$OBS_RC" -ne 0 ]; then
  echo "===== OBS FAIL → ROLLBACK ====="
  bash /tmp/_p0e_rollback_global.sh
  exit $OBS_RC
fi

echo "===== KEEP PG_FIRST=true (GLOBAL ENABLE HOLDING) ====="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
exit 0
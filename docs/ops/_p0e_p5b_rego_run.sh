set -euo pipefail
cd /srv/atmr
cp /tmp/_p0e_phase2_enable.sh /tmp/_p0e_enable_133.sh
cp /tmp/_p0e_phase2_rollback.sh /tmp/_p0e_rollback_133.sh
chmod +x /tmp/_p0e_enable_133.sh /tmp/_p0e_rollback_133.sh
docker cp /tmp/_p0e_p5b_rego_canary.py atmr-backend-1:/tmp/ 2>/dev/null || true

echo "===== ENABLE ====="
bash /tmp/_p0e_enable_133.sh
echo "===== CANARY ====="
docker cp /tmp/_p0e_p5b_rego_canary.py atmr-backend-1:/tmp/
set +e
docker exec -e P0E_PIN_SESSION=trk_sess_1786977672739_0rzte5pe \
  -e P0E_DRIVER_ID=20135 -e P0E_WAIT_SEC=90 \
  atmr-backend-1 python /tmp/_p0e_p5b_rego_canary.py
CANARY_RC=$?
set -e
echo "CANARY_RC=$CANARY_RC"

echo "===== ROLLBACK PG_FIRST=false (toujours) ====="
bash /tmp/_p0e_rollback_133.sh
echo "FINAL_FLAGS"
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED TRACKING_PERSIST_WITH_OUTBOX
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED TRACKING_PERSIST_WITH_OUTBOX
exit $CANARY_RC
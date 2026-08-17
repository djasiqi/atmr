#!/usr/bin/env bash
# Restore TRACKING_PERSIST_WITH_OUTBOX=true (pre-deploy behavior). PG_FIRST stays false.
set -euo pipefail
cd /srv/atmr

if grep -q "^TRACKING_PERSIST_WITH_OUTBOX=" .env.production; then
  sed -i "s|^TRACKING_PERSIST_WITH_OUTBOX=.*|TRACKING_PERSIST_WITH_OUTBOX=true|" .env.production
else
  echo "TRACKING_PERSIST_WITH_OUTBOX=true" >> .env.production
fi

# Ensure PG_FIRST still false
if grep -q "^TRACKING_PG_FIRST_CANONICAL_ENABLED=" .env.production; then
  sed -i "s|^TRACKING_PG_FIRST_CANONICAL_ENABLED=.*|TRACKING_PG_FIRST_CANONICAL_ENABLED=false|" .env.production
else
  echo "TRACKING_PG_FIRST_CANONICAL_ENABLED=false" >> .env.production
fi

echo "VERIFY:"
grep -E "^(TRACKING_PERSIST_WITH_OUTBOX|TRACKING_PG_FIRST_CANONICAL_ENABLED|DOCKER_TAG)=" .env.production

docker compose -f docker-compose.production.yml --profile kafka --env-file .env.production \
  up -d --no-deps --force-recreate tracking-kafka-consumer

for i in $(seq 1 40); do
  CH=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' atmr-tracking-kafka-consumer-1)
  echo "t=${i} consumer=${CH}"
  if [ "${CH}" = "healthy" ] || [ "${CH}" = "running" ]; then
    break
  fi
  sleep 2
done

docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
docker cp /tmp/_p0e_check_consumer_flags.py atmr-tracking-kafka-consumer-1:/tmp/_p0e_check_consumer_flags.py
docker exec -w /app atmr-tracking-kafka-consumer-1 python /tmp/_p0e_check_consumer_flags.py
echo OUTBOX_RESTORE_DONE

#!/usr/bin/env bash
# Vérif post-rollback avant pm clear
set -euo pipefail
test "$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)" = "true"
test "$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)" = "healthy"
test "$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)" = "healthy"
echo "backend_pg_first=$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)"
echo "consumer_pg_first=$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED)"
echo "outbox=$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)"
echo "backend=$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)"
echo "consumer=$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)"
echo POST_ROLLBACK_VERIFY_PASS

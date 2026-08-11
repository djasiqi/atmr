#!/usr/bin/env bash
# Phase 2 — autopsie seq=3 100% read-only.
set -euo pipefail
cd /srv/atmr

DRIVER_ID=3
SESSION_ID='trk_sess_1786447105637_ivp6dqaq'
GEN=1204

psql() {
  docker compose -p atmr --env-file .env.production -f docker-compose.production.yml \
    exec -T postgres psql -U atmr -d atmr -v ON_ERROR_STOP=1 "$@"
}

echo "======== watermark / gaps ========"
psql -c "
SELECT tracking_session_id, session_generation,
       contiguous_persisted_through, max_seen_sequence,
       first_seen_at, last_seen_at, closed_at
FROM tracking_session_state
WHERE driver_id = ${DRIVER_ID}
  AND tracking_session_id = '${SESSION_ID}';
"

psql -c "
SELECT sequence_from, sequence_to, detected_at, resolved_at, session_generation
FROM tracking_sequence_gaps
WHERE driver_id = ${DRIVER_ID}
  AND tracking_session_id = '${SESSION_ID}'
ORDER BY sequence_from;
" 2>/dev/null || psql -c "
SELECT sequence_from, sequence_to, detected_at, resolved_at
FROM tracking_sequence_gaps
WHERE driver_id = ${DRIVER_ID}
  AND tracking_session_id = '${SESSION_ID}'
ORDER BY sequence_from;
"

echo "======== ledger sequences ========"
psql -c "
SELECT sequence_id, location_event_id, recorded_at, source, received_at
FROM tracking_ingest_events
WHERE driver_id = ${DRIVER_ID}
  AND tracking_session_id = '${SESSION_ID}'
  AND session_generation = ${GEN}
ORDER BY sequence_id;
"

echo "======== missing seq check 1..max ========"
psql -c "
WITH bounds AS (
  SELECT COALESCE(MAX(sequence_id),0) AS mx
  FROM tracking_ingest_events
  WHERE driver_id = ${DRIVER_ID}
    AND tracking_session_id = '${SESSION_ID}'
    AND session_generation = ${GEN}
),
series AS (
  SELECT generate_series(1, GREATEST((SELECT mx FROM bounds),1)) AS sequence_id
)
SELECT s.sequence_id AS missing_sequence
FROM series s
LEFT JOIN tracking_ingest_events e
  ON e.driver_id = ${DRIVER_ID}
 AND e.tracking_session_id = '${SESSION_ID}'
 AND e.session_generation = ${GEN}
 AND e.sequence_id = s.sequence_id
WHERE e.location_event_id IS NULL
  AND (SELECT mx FROM bounds) > 0
ORDER BY s.sequence_id;
"

echo "======== DLE for session ========"
psql -c "
SELECT sequence_id, location_event_id, recorded_at, source
FROM driver_location_events
WHERE driver_id = ${DRIVER_ID}
  AND tracking_session_id = '${SESSION_ID}'
  AND session_generation = ${GEN}
ORDER BY sequence_id
LIMIT 100;
"

echo "======== outbox for session events ========"
psql -c "
SELECT o.id, o.location_event_id, o.published_at, o.created_at
FROM tracking_event_outbox o
JOIN tracking_ingest_events e
  ON e.location_event_id = o.location_event_id
 AND e.driver_id = o.driver_id
WHERE e.driver_id = ${DRIVER_ID}
  AND e.tracking_session_id = '${SESSION_ID}'
  AND e.session_generation = ${GEN}
ORDER BY o.created_at;
" 2>/dev/null || psql -c "
SELECT column_name FROM information_schema.columns
WHERE table_name = 'tracking_event_outbox' ORDER BY ordinal_position;
"

echo "======== specifically sequence_id = 3 ========"
psql -c "
SELECT 'ledger' AS src, location_event_id::text, recorded_at::text, source::text
FROM tracking_ingest_events
WHERE driver_id = ${DRIVER_ID} AND tracking_session_id = '${SESSION_ID}'
  AND session_generation = ${GEN} AND sequence_id = 3
UNION ALL
SELECT 'dle', location_event_id::text, recorded_at::text, source::text
FROM driver_location_events
WHERE driver_id = ${DRIVER_ID} AND tracking_session_id = '${SESSION_ID}'
  AND session_generation = ${GEN} AND sequence_id = 3;
"

echo "======== other generations same session id ========"
psql -c "
SELECT session_generation, COUNT(*), MIN(sequence_id), MAX(sequence_id)
FROM tracking_ingest_events
WHERE driver_id = ${DRIVER_ID} AND tracking_session_id = '${SESSION_ID}'
GROUP BY session_generation
ORDER BY session_generation;
"

echo "======== access logs window (best effort) ========"
cid="$(docker ps -q --filter name=atmr-backend | head -n1 || true)"
if [[ -n "${cid}" ]]; then
  docker logs --since 72h "${cid}" 2>&1 \
    | grep -E 'trk_sess_1786447105637_ivp6dqaq|/driver/me/location' \
    | grep -E '11:19:|sequence|401|403|429|202|672|393' \
    | tail -n 60 || echo "(pas de match access logs rétention)"
else
  echo "backend cid absent"
fi

echo "PHASE2_SQL_DONE"
